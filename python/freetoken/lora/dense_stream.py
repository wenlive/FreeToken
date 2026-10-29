"""Bounded eager inference: finalized packed layer bytes share one CUDA slot."""

from __future__ import annotations

import hashlib
import types

import torch

from freetoken.layers.base import BaseOP, StateLessOP
from freetoken.layers.quantization import finalize_quant


def _tensor_attributes(root):
    seen = set()
    def walk(op, prefix):
        # RoPE tables are shared generated device caches, outside checkpoint state.
        if isinstance(op, StateLessOP):
            return
        if id(op) in seen:
            return
        seen.add(id(op))
        for name, value in vars(op).items():
            if isinstance(value, torch.Tensor):
                yield op, name, f"{prefix}{name}", value
            elif isinstance(value, BaseOP):
                yield from walk(value, f"{prefix}{name}.")
    yield from walk(root, "")


def _move(root, device):
    for op, name, _, value in list(_tensor_attributes(root)):
        setattr(op, name, value.to(device=device).contiguous())


class DenseLayerStreamer:
    """Copy on the compute stream so slot reuse follows its last reader in order."""

    def __init__(self, layers, device):
        self.device = torch.device(device)
        self.stream = torch.cuda.current_stream(self.device)
        self.banks, self.fields = [], []
        self.copy_count = self.copy_bytes = 0
        self._layers = list(layers)
        for layer in self._layers:
            if hasattr(layer.mlp, "experts") or hasattr(layer.mlp, "shared_expert"):
                raise ValueError("Dense layer streaming cannot own MoE layers")
        for layer in self._layers:
            _move(layer, self.device)
            finalize_quant(layer)
            entries, total = [], 0
            for op, name, path, value in _tensor_attributes(layer):
                if value.device.type != "cuda":
                    raise ValueError(f"Unmaterialized dense layer tensor: {path}")
                total = (total + 15) // 16 * 16
                entries.append((op, name, path, total, value.shape, value.dtype, value.numel() * value.element_size()))
                total += entries[-1][-1]
            bank = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            for op, name, _, offset, _, _, size in entries:
                value = getattr(op, name)
                bank[offset:offset + size].copy_(value.reshape(-1).view(torch.uint8), non_blocking=True)
            self.stream.synchronize()
            # Release this layer's device storage before finalizing the next layer.
            for op, name, _, offset, shape, dtype, size in entries:
                setattr(op, name, bank[offset:offset + size].view(dtype).view(shape))
            self.banks.append(bank)
            self.fields.append(entries)
        self.slot = torch.empty(max(bank.numel() for bank in self.banks), dtype=torch.uint8, device=self.device)
        for index, (layer, entries) in enumerate(zip(self._layers, self.fields)):
            for op, name, _, offset, shape, dtype, size in entries:
                setattr(op, name, self.slot[offset:offset + size].view(dtype).view(shape))
            original = layer.forward
            manager = self
            def forward(instance, *args, _index=index, _original=original, **kwargs):
                manager.acquire(_index)
                return _original(*args, **kwargs)
            layer.forward = types.MethodType(forward, layer)

    def acquire(self, index):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dense layer streaming requires eager execution, without CUDA graphs")
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("Dense layer streaming must stay on its construction stream")
        bank = self.banks[index]
        self.slot[:bank.numel()].copy_(bank, non_blocking=True)
        self.copy_count += 1
        self.copy_bytes += bank.numel()

    def inventory(self):
        return {"layers": len(self.banks), "packed_host_payload_bytes": sum(bank.numel() for bank in self.banks),
                "packed_host_all_pinned": all(bank.is_pinned() for bank in self.banks),
                "gpu_slot_bytes": self.slot.numel(), "slot_count": 1, "copies": self.copy_count,
                "copy_bytes": self.copy_bytes, "cuda_graphs": False}

    def hashes(self):
        return [hashlib.sha256(bank.numpy().tobytes()).hexdigest() for bank in self.banks]


def load_streamed_dense_weights(model, config, device, materialize):
    """Use the existing checkpoint reader and quant finalizers one layer at a time."""
    from freetoken.models import load_weight
    from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM

    if (not isinstance(model, Qwen3_5ForCausalLM) or config.model_config.is_moe
            or config.tp_info.size != 1 or config.use_dummy_weight or config.active_encoders
            or config.mtp_runtime_path is not None or config.gguf_embedding_host
            or config.cuda_graph_bs or config.cuda_graph_max_bs != 0):
        raise ValueError("Dense streaming requires real single-GPU text-only Qwen dense with --cuda-graph-max-bs 0")
    cpu = torch.device("cpu")
    state = materialize(model.state_dict(), load_weight(config.model_path, cpu, include_moe_experts=False,
                        include_vision=False), device=cpu)
    model.load_state_dict(state)
    streamer = DenseLayerStreamer(model.model.layers.op_list, device)
    _move(model.model.embed_tokens, device)
    _move(model.model.norm, device)
    _move(model.lm_head, device)
    finalize_quant(model.lm_head)
    model._dense_streamer = streamer
    return streamer.inventory()
