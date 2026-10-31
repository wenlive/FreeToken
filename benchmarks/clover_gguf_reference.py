"""Official HF Clover operators with an independent GGUF expert-dot reference.

GGML weights decode with gguf-py's NumPy implementation. The expert linear
operation emulates Q8_1 activation blocks in float32; no production CUDA GGUF
kernel is used by the reference. Dense loader mapping is shared with production
and qualified separately against original BF16 tensors and explicit head labels.
"""

from collections import OrderedDict
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as functional

from clover_hf_reference import official_modules
from freetoken.models.clover_moe.gguf import iter_hf_gguf_weights, parse_gguf_config
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
        if kind not in (12, 23):
            raise ValueError(f"Unsupported GGUF reference expert type {kind}")
        output = functional.linear(q8_1_activations(value), weight.float(), bias)
        return output.to(value.dtype)


class GGUFExpertStorage:
    def __init__(self, checkpoint, *, device="cuda", cache_bytes=6 << 30):
        self.path = str(checkpoint)
        self.device, self.limit = device, cache_bytes
        self.tensors = {
            tensor.name: tensor
            for tensor in iter_gguf_tensors(self.path)
            if "_exps.weight" in tensor.name
        }
        self.cache = OrderedDict()
        self.bytes = self.reads = self.hits = 0

    def projection(self, layer, expert, projection):
        key = layer, int(expert), projection
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.reads += 1
        tensor = self.tensors[f"blk.{layer}.ffn_{projection}_exps.weight"]
        raw = tensor._raw.reshape(tensor.shape[0], tensor.shape[1], -1)[int(expert)]
        decoded = gguf.dequantize(raw, gguf.GGMLQuantizationType(tensor.ggml_type))
        value = torch.from_numpy(np.array(decoded, copy=True)).to(self.device)
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
    for name, value in iter_hf_gguf_weights(str(checkpoint), config):
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
    if loaded != set(expected) or any(p.is_meta for p in model.parameters()):
        raise ValueError("Incomplete GGUF reference dense weights")
    # nn.Linear dense operators stay unchanged. Only the expert F.linear sites
    # in the official Python module see the independent Q8_1 dot reference.
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
