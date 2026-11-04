"""Clover GGUF adapter with native packed dense projections and expert banks.

Layout reference: ggml-org/llama.cpp@4838ffb0c8ade92f34d9157deea38d92758ab824,
conversion/clover.py and conversion/qwen.py (MIT). This adapter reverses the
exported cyclic V-head order; causal/preview operators remain FreeToken's.
"""

from dataclasses import replace
import re
from types import SimpleNamespace

import torch

from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.models.qwen3_5_moe.config import parse_config as parse_qwen
from freetoken.models.qwen3_5_moe.weight import _is_gemma_norm

from .model import CloverMoeModelLM as BaseCloverModel

EXPERT_FORMAT = "clover_gguf"
SUPPORTED_EXPERT_TYPES = {2, 8, 12, 14, 23}
_EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
_SCALARS = {
    "ssm_a": "A_log",
    "ssm_dt.bias": "dt_bias",
    "ssm_bd_history.bias": "bd_softmax_history_bias",
    "ssm_bd_history.weight": "bd_softmax_history_c",
    "ssm_bd_logit_scale.weight": "bd_softmax_logit_scale",
}
_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_conv1d.weight_right": "linear_attn.conv_right_weight",
    **{key: "linear_attn." + value for key, value in _SCALARS.items()},
}


def _require_metadata(shim, key):
    name = "clover_moe." + key
    if name not in shim.metadata:
        raise ValueError(f"Clover GGUF is missing {name}")
    return shim.metadata[name]


def _validate_experts(path, config):
    expected = {
        (layer, role)
        for layer in range(config.num_layers)
        for role in ("gate", "up", "down")
    }
    seen = set()
    descriptors = {}
    for tensor in iter_gguf_tensors(path):
        match = _EXPERT.fullmatch(tensor.name)
        if not match:
            continue
        layer, role = int(match[1]), match[2]
        key = layer, role
        if key not in expected or key in seen:
            raise ValueError(
                f"Unexpected or duplicate Clover expert tensor {tensor.name}"
            )
        h, i, e = config.hidden_size, config.moe_intermediate_size, config.num_experts
        shape = (e, h, i) if role == "down" else (e, i, h)
        if tensor.shape != shape or tensor.ggml_type not in SUPPORTED_EXPERT_TYPES:
            raise ValueError(
                f"{tensor.name}: expected shape {shape}, supported packed GGML type; "
                f"got {tensor.shape}, type {tensor.ggml_type}"
            )
        seen.add(key)
        descriptors[key] = (tensor.ggml_type, tensor.row_bytes)
    if seen != expected:
        raise ValueError(
            f"Missing Clover GGUF expert tensors: {sorted(expected - seen)[:8]}"
        )
    for layer in range(config.num_layers):
        if descriptors[layer, "gate"] != descriptors[layer, "up"]:
            raise ValueError(f"Layer {layer} gate/up expert types differ")
    return descriptors


def parse_gguf_config(shim):
    def g(key):
        return _require_metadata(shim, key)

    layers = int(g("block_count"))
    recurrent = g("attention.recurrent_layers")
    if len(recurrent) != layers or any(type(item) is not bool for item in recurrent):
        raise ValueError("Clover GGUF recurrent_layers must contain one bool per layer")
    nk, nv = int(g("ssm.group_count")), int(g("ssm.time_step_rank"))
    inner, dk = int(g("ssm.inner_size")), int(g("ssm.state_size"))
    head = int(g("attention.key_length"))
    if nk <= 0 or nv <= 0 or nv % nk or inner % nv:
        raise ValueError("Invalid Clover GDN head geometry")
    hf = SimpleNamespace(
        model_type="clover_moe",
        architectures=shim.architectures,
        hidden_size=int(g("embedding_length")),
        vocab_size=shim.vocab_size,
        num_hidden_layers=layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=head,
        max_position_embeddings=int(g("context_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        tie_word_embeddings=shim.tie_word_embeddings,
        linear_num_key_heads=nk,
        linear_num_value_heads=nv,
        linear_key_head_dim=dk,
        linear_value_head_dim=inner // nv,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")),
        layer_types=["linear_attention" if r else "full_attention" for r in recurrent],
        rope_parameters=dict(
            rope_type="default",
            rope_theta=float(g("rope.freq_base")),
            partial_rotary_factor=int(g("rope.dimension_count")) / head,
        ),
    )
    config = replace(
        parse_qwen(hf), expert_quant=EXPERT_FORMAT, moe_weight_format=EXPERT_FORMAT
    )
    experts = _validate_experts(shim.model_path, config)
    from .packed import CloverDenseGroups

    groups = CloverDenseGroups()
    dense = {}
    for tensor in iter_gguf_tensors(shim.model_path):
        if tensor.ggml_type not in (2, 8, 12, 14) or _EXPERT.fullmatch(tensor.name):
            continue
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
        if match is None or match[2] not in _LAYER:
            continue
        module = f"model.layers.{int(match[1])}.{_LAYER[match[2]]}".removesuffix(".weight")
        target, _, _ = groups.target(module)
        rows, cols = tensor.shape
        previous = dense.get(target)
        if previous is not None:
            if previous[1:] != (cols, tensor.ggml_type, tensor.row_bytes):
                raise ValueError(f"Incompatible packed projection group {target}")
            rows += previous[0]
        dense[target] = (rows, cols, tensor.ggml_type, tensor.row_bytes)
    return replace(config, expert_source=dict(dense=dense,
        types=[(experts[l, "gate"][0], experts[l, "down"][0]) for l in range(layers)],
        row_bytes=[(experts[l, "gate"][1], experts[l, "down"][1]) for l in range(layers)]))


class CloverMoeModelLM(BaseCloverModel):
    def __init__(self, config):
        super().__init__(config)
        from .packed import install_packed_linears

        install_packed_linears(self, config)


def restore_v_heads(value, axis, nk, nv, head_dim):
    """Inverse of the exporter's [K, V/K, D] -> [V/K, K, D] permutation."""
    axis %= value.ndim
    if value.shape[axis] != nv * head_dim:
        raise ValueError("Clover V-head axis does not match the declared geometry")
    shape = list(value.shape)
    split = shape[:axis] + [nv // nk, nk, head_dim] + shape[axis + 1 :]
    return value.reshape(split).transpose(axis, axis + 1).reshape(shape).contiguous()


def _restore_dense(name, value, config, *, dtype=torch.bfloat16):
    group = config.linear_attention_group()
    nk, nv, dk, dv = (
        group.num_key_heads,
        group.num_value_heads,
        group.key_head_dim,
        group.value_head_dim,
    )
    if ".linear_attn." in name:
        suffix = name.split(".linear_attn.", 1)[1]
        if suffix in ("in_proj_qkv.weight", "conv1d.weight", "conv_right_weight"):
            boundary = 2 * nk * dk
            value = torch.cat(
                (value[:boundary], restore_v_heads(value[boundary:], 0, nk, nv, dv))
            )
            if suffix in ("conv1d.weight", "conv_right_weight"):
                value = value.unsqueeze(1)
        elif suffix == "in_proj_z.weight":
            value = restore_v_heads(value, 0, nk, nv, dv)
        elif suffix in ("in_proj_a.weight", "in_proj_b.weight", *_SCALARS.values()):
            value = restore_v_heads(value, 0, nk, nv, 1)
        elif suffix == "out_proj.weight":
            value = restore_v_heads(value, 1, nk, nv, dv)
        # Clover exports raw A_log, not Qwen's -exp(A_log).
    if _is_gemma_norm(name):
        value = value.float() - 1.0
    if name.endswith(".shared_expert_gate.weight"):
        value = value.reshape(1, config.hidden_size)
    return value.to(dtype)


def iter_gguf_weights(
    model_path, device, *, include_moe_experts, include_non_moe, include_vision=True
):
    if include_moe_experts or not include_non_moe:
        raise ValueError("Clover GGUF requires native packed expert offload")
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    from .packed import CloverDenseGroups
    import gguf
    import numpy as np

    reader, pending = CloverDenseGroups(), {}
    globals_ = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight",
                "output_norm.weight": "model.norm.weight"}
    for tensor in iter_gguf_tensors(model_path):
        if _EXPERT.fullmatch(tensor.name):
            continue
        name = globals_.get(tensor.name)
        if name is None:
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
            if match is None or match[2] not in _LAYER:
                raise ValueError(f"Unmapped Clover tensor {tensor.name}")
            name = f"model.layers.{int(match[1])}.{_LAYER[match[2]]}"
        target, index, count = reader.target(name.removesuffix(".weight"))
        if target in config.expert_source["dense"]:
            value = tensor.packed().clone()
            group = config.linear_attention_group()
            nk, nv, dv = group.num_key_heads, group.num_value_heads, group.value_head_dim
            if name.endswith(".linear_attn.in_proj_qkv.weight"):
                boundary = 2 * nk * group.key_head_dim
                value = torch.cat((value[:boundary], restore_v_heads(value[boundary:], 0, nk, nv, dv)))
            elif name.endswith(".linear_attn.in_proj_z.weight"):
                value = restore_v_heads(value, 0, nk, nv, dv)
            elif name.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
                value = restore_v_heads(value, 0, nk, nv, 1)
            elif name.endswith(".linear_attn.out_proj.weight"):
                block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(tensor.ggml_type)]
                if dv % block:
                    raise ValueError("Packed GDN output permutation crosses quantization blocks")
                value = restore_v_heads(value, 1, nk, nv, dv // block * size)
            parts = pending.setdefault(target, {})
            parts[index] = value
            if len(parts) == count:
                yield target + ".weight", torch.cat([parts[i] for i in range(count)]).to(device)
                del pending[target]
        else:
            values = gguf.dequantize(tensor._raw, gguf.GGMLQuantizationType(tensor.ggml_type))
            value = _restore_dense(name, torch.from_numpy(np.array(values, copy=True)).reshape(tensor.shape), config)
            # BF16 b/a must follow the split projection used by the packed model.
            if target.endswith(".in_proj_ba"):
                parts = pending.setdefault(target, {})
                parts[index] = value
                if len(parts) == count:
                    yield target + ".weight", torch.cat([parts[i] for i in range(count)]).to(device)
                    del pending[target]
            else:
                yield name, value.to(device)
    if pending:
        raise ValueError(f"Incomplete Clover projections: {list(pending)}")
    from freetoken.models.gguf.reader import _reader
    _reader.cache_clear()


def load_packed_expert_sources(model_path, config, *, layer_sink=None, dummy=False):
    from freetoken.moe.host_banks import (
        LayerCompletionTracker,
        PinPipeline,
        alloc_layer_banks,
    )

    if dummy:
        raise NotImplementedError(
            "Clover GGUF requires real packed weights; dummy banks are not defined"
        )
    descriptors = _validate_experts(model_path, config)
    e, h, i = config.num_experts, config.hidden_size, config.moe_intermediate_size
    if h % 256 or i % 256:
        raise ValueError("Mixed Clover GGUF expert widths must be multiples of 256")
    gate_bytes = max(v[1] for (l, role), v in descriptors.items() if role == "gate")
    down_bytes = max(v[1] for (l, role), v in descriptors.items() if role == "down")
    hb = alloc_layer_banks(
        {
            "gate_up": ((e, 2 * i, gate_bytes), torch.uint8),
            "down": ((e, h, down_bytes), torch.uint8),
        },
        config.num_layers,
    )
    banks = {
        name: [bank.tensor for bank in per_layer] for name, per_layer in hb.items()
    }

    def fill(sink):
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        from freetoken.models.gguf.reader import _reader
        # Only the tensor directory is consulted; readinto never faults payload
        # into a persistent file mapping alongside the 32 GiB pinned banks.
        directory = _reader(model_path)
        by_name = {t.name: t for t in directory.tensors}
        for tensor in iter_gguf_tensors(model_path):
            match = _EXPERT.fullmatch(tensor.name)
            if match is None:
                continue
            layer, role = int(match[1]), match[2]
            bank = banks["down" if role == "down" else "gate_up"][layer]
            rb = descriptors[layer, role][1]
            rows = h if role == "down" else i
            offset = i * rb if role == "up" else 0
            destination = bank.as_strided((e, rows * rb), (bank.stride(0), 1), offset)
            file_tensor = by_name[tensor.name]
            with open(model_path, "rb", buffering=0) as stream:
                stream.seek(file_tensor.data_offset)
                if destination.is_contiguous():
                    view = memoryview(destination.numpy()).cast("B")
                    while view:
                        n = stream.readinto(view)
                        if not n:
                            raise ValueError(f"Truncated Clover expert {tensor.name}")
                        view = view[n:]
                else:
                    for expert in destination:
                        view = memoryview(expert.numpy()).cast("B")
                        while view:
                            n = stream.readinto(view)
                            if not n:
                                raise ValueError(f"Truncated Clover expert {tensor.name}")
                            view = view[n:]
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        fill(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            fill(pins)
    else:
        fill(None)
    from freetoken.models.gguf.reader import _reader
    _reader.cache_clear()
    return banks
