"""Official HF Clover operators with an independent GGUF expert-dot reference.

GGML weights use an independent eager-Torch decoder qualified bitwise against
gguf-py's NumPy implementation; NumPy remains the CPU/fallback decoder. The
expert linear emulates Q8_1 activation blocks in float32 without production
CUDA GGUF kernels. Dense loader mapping is shared with production and qualified
separately against original BF16 tensors and explicit head labels.
"""

from collections import OrderedDict
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as functional

from clover_gguf_decode import decode_expert
from clover_hf_reference import official_modules
from freetoken.models.clover_moe.gguf import parse_gguf_config
from clover.oracles import iter_hf_gguf_weights
from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.utils import cached_load_hf_config


def q8_1_activations(value):
    """GGML per32 max/127, away-from-zero integer rounding, FP16 stored scale."""
    shape = value.shape
    if shape[-1] % 32:
        raise ValueError("GGML Q8_1 activation width must be divisible by32")
    blocks = value.float().reshape(*shape[:-1], -1, 32)
    scale = blocks.abs().amax(-1, keepdim=True) / 127.0
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = blocks / safe
    codes = normalized.sign() * (normalized.abs() + 0.5).floor()
    return (codes * scale.half().float()).reshape(shape)


class ReferenceFunctional:
    """Limit altered expert arithmetic to this official module's F.linear calls."""

    def __getattr__(self, name):
        return getattr(functional, name)

    def linear(self, value, weight, bias=None):
        kind = getattr(weight, "_clover_ggml_type", None)
        if kind is None:
            return functional.linear(value, weight, bias)
        if kind not in (2, 8, 12, 14, 23):
            raise ValueError(f"Unsupported GGUF reference expert type {kind}")
        output = functional.linear(q8_1_activations(value), weight.float(), bias)
        return output.to(value.dtype)


class ReferenceQ8Linear(torch.nn.Module):
    """Independent Q8_0 weights and Q8_1 activation arithmetic for dense operators."""
    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.bias = module.bias
        self.in_features, self.out_features = module.in_features, module.out_features

    def forward(self, value):
        return functional.linear(q8_1_activations(value), self.weight, self.bias).to(value.dtype)


class GGUFExpertStorage:
    def __init__(
        self,
        checkpoint,
        *,
        device="cuda",
        cache_bytes=6 << 30,
        cpu_cache_bytes=16 << 30,
        decoder="torch",
    ):
        if decoder not in ("torch", "numpy"):
            raise ValueError("Independent expert decoder must be torch or numpy")
        self.decoder = decoder if str(device).startswith("cuda") else "numpy"
        self.path = str(checkpoint)
        self.device, self.limit = device, cache_bytes
        self.tensors = {
            tensor.name: tensor
            for tensor in iter_gguf_tensors(self.path)
            if "_exps.weight" in tensor.name
        }
        self.cache = OrderedDict()
        self.cpu_cache = OrderedDict()
        self.cpu_limit = cpu_cache_bytes if self.decoder == "numpy" else 0
        self.cpu_bytes = 0
        self.cpu_hits = 0
        self.bytes = self.reads = self.hits = 0

    def projection(self, layer, expert, projection):
        key = layer, int(expert), projection
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.reads += 1
        tensor = self.tensors[f"blk.{layer}.ffn_{projection}_exps.weight"]
        if self.decoder == "torch":
            raw = tensor._raw.reshape(tensor.shape[0], tensor.shape[1], -1)[int(expert)]
            value = decode_expert(raw, tensor.ggml_type, device=self.device)
        elif key in self.cpu_cache:
            self.cpu_hits += 1
            self.cpu_cache.move_to_end(key)
            host = self.cpu_cache[key]
            value = host.to(self.device)
        else:
            raw = tensor._raw.reshape(tensor.shape[0], tensor.shape[1], -1)[int(expert)]
            decoded = gguf.dequantize(raw, gguf.GGMLQuantizationType(tensor.ggml_type))
            host = torch.from_numpy(np.array(decoded, copy=True))
            size = host.numel() * host.element_size()
            while self.cpu_cache and self.cpu_bytes + size > self.cpu_limit:
                _, evicted = self.cpu_cache.popitem(last=False)
                self.cpu_bytes -= evicted.numel() * evicted.element_size()
            if size <= self.cpu_limit:
                self.cpu_cache[key] = host
                self.cpu_bytes += size
            value = host.to(self.device)
        value._clover_ggml_type = tensor.ggml_type
        count = value.numel() * value.element_size()
        while self.cache and self.bytes + count > self.limit:
            _, evicted = self.cache.popitem(last=False)
            self.bytes -= evicted.numel() * evicted.element_size()
        if count <= self.limit:
            self.cache[key] = value
            self.bytes += count
        return value


class GGUFExpertIndex:
    def __init__(self, storage, layer, projection):
        self.storage, self.layer, self.projection = storage, layer, projection

    def __getitem__(self, expert):
        if self.projection != "gate_up":
            return self.storage.projection(self.layer, expert, self.projection)
        gate, up = [
            self.storage.projection(self.layer, expert, role) for role in ("gate", "up")
        ]
        value = torch.cat((gate, up), dim=0)
        value._clover_ggml_type = gate._clover_ggml_type
        return value


def load_gguf_official(source, checkpoint, *, device="cuda", cache_bytes=6 << 30):
    torch.set_float32_matmul_precision("highest")
    model_code, config_code, _ = official_modules(Path(source))
    raw = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    raw.pop("quantization_config", None)
    with torch.device("meta"):
        model = model_code.CloverMoeModelLM(config_code.CloverMoeConfig(**raw))
    storage = GGUFExpertStorage(checkpoint, device=device, cache_bytes=cache_bytes)
    for index, layer in enumerate(model.model.layers):
        del layer.mlp.experts.gate_up_proj
        del layer.mlp.experts.down_proj
        layer.mlp.experts.gate_up_proj = GGUFExpertIndex(storage, index, "gate_up")
        layer.mlp.experts.down_proj = GGUFExpertIndex(storage, index, "down")
    expected = dict(model.named_parameters())
    config = parse_gguf_config(cached_load_hf_config(str(checkpoint)))
    loaded = set()
    q8_dense = []
    for name, value in iter_hf_gguf_weights(str(checkpoint), config, preserve_q8=True):
        if (
            name not in expected
            or name in loaded
            or value.shape != expected[name].shape
        ):
            raise ValueError(f"GGUF reference parameter mismatch: {name}")
        parent, leaf = name.rsplit(".", 1)
        setattr(
            model.get_submodule(parent),
            leaf,
            torch.nn.Parameter(value.to(device), requires_grad=False),
        )
        loaded.add(name)
        if value.dtype == torch.float32 and isinstance(model.get_submodule(parent), torch.nn.Linear):
            q8_dense.append(parent)
    if loaded != set(expected) or any(p.is_meta for p in model.parameters()):
        raise ValueError("Incomplete GGUF reference dense weights")
    for path in q8_dense:
        parent, leaf = path.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, ReferenceQ8Linear(model.get_submodule(path)))
    # The model's HF control flow is unchanged. Quantized dense/expert arithmetic
    # is a mathematical oracle, independent of serving GGML CUDA kernels.
    reference = ReferenceFunctional()
    original_fp32_linear = model_code.linear_fp32_output

    def fp32_linear(value, weight):
        if getattr(weight, "_clover_ggml_type", None) is not None:
            # Official's BF16 fast path calls torch.mm(out_dtype=FP32) directly.
            # The decoded oracle weights are FP32: route that expert-only site
            # through the same independent Q8_1 reference, preserving FP32 down.
            return reference.linear(value.float(), weight)
        return original_fp32_linear(value, weight)

    model_code.F = reference
    model_code.linear_fp32_output = fp32_linear
    model.eval()
    return model, storage
