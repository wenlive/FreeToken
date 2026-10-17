"""Flash Next 的完整 GGUF 目录、压缩行读取和显式张量映射。

格式依据冻结 Strata 6674a006 的 gguf_reader.hpp；不依赖它的运行时。
GGUF 的维度按最内层在前记录，本模块保留该顺序，torch_shape 显式反转。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re


from freetoken.checkpoint.gguf_reader import (
    GGUF_TYPES,
    read_exact as _read,
    unpack as _unpack,
    read_string as _string,
    read_value as _value,
)


@dataclass(frozen=True)
class GGUFTensor:
    path: Path
    name: str
    shape: tuple[int, ...]
    type_id: int
    start: int

    @property
    def type_name(self):
        return GGUF_TYPES[self.type_id][0]

    @property
    def torch_shape(self):
        return self.shape[::-1]

    @property
    def row_bytes(self):
        _, block, size = GGUF_TYPES[self.type_id]
        return self.shape[0] // block * size

    @property
    def nbytes(self):
        return math.prod(self.shape[1:]) * self.row_bytes

    def read_rows(self, first, count):
        """仅读取请求的连续压缩行；embedding/PLE 无需整表解量化。"""
        rows = math.prod(self.shape[1:])
        if first < 0 or count < 0 or first + count > rows:
            raise ValueError(f"{self.name}: invalid row range {first}:{first + count}")
        with self.path.open("rb", buffering=0) as stream:
            stream.seek(self.start + first * self.row_bytes)
            return _read(stream, count * self.row_bytes)


class GGUFShard:
    def __init__(self, path):
        """解析目录并检查所有张量边界、块宽度和重叠，不只检查专家。"""
        self.path = Path(path).resolve()
        self.metadata, self.tensors = {}, {}
        with self.path.open("rb") as stream:
            magic, version, tensors, values = _unpack(stream, "IIQQ")
            if magic != 0x46554747 or version != 3 or tensors > 100_000 or values > 100_000:
                raise ValueError("invalid GGUF v3 header")
            for _ in range(values):
                key = _string(stream)
                if key in self.metadata:
                    raise ValueError(f"duplicate GGUF metadata {key}")
                value = _value(stream, keep=not key.startswith("tokenizer."))
                self.metadata[key] = value
            directory = []
            for _ in range(tensors):
                name = _string(stream)
                dims, = _unpack(stream, "I")
                if not 1 <= dims <= 4:
                    raise ValueError(f"{name}: invalid GGUF rank")
                shape = _unpack(stream, "Q" * dims)
                kind, offset = _unpack(stream, "IQ")
                directory.append((name, shape, kind, offset))
            alignment = self.metadata.get("general.alignment", 32)
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError("invalid GGUF alignment")
            self.data_start = (stream.tell() + alignment - 1) // alignment * alignment
        source_size, spans = self.path.stat().st_size, []
        for name, shape, kind, offset in directory:
            if name in self.tensors:
                raise ValueError(f"duplicate GGUF tensor {name}")
            if kind not in GGUF_TYPES:
                raise ValueError(f"{name}: unsupported GGUF tensor type {kind}")
            if any(d <= 0 for d in shape) or shape[0] % GGUF_TYPES[kind][1]:
                raise ValueError(f"{name}: incompatible GGUF block geometry {shape}")
            tensor = GGUFTensor(self.path, name, shape, kind, self.data_start + offset)
            if offset % alignment or tensor.start + tensor.nbytes > source_size:
                raise ValueError(f"{name}: out-of-bounds or misaligned GGUF tensor")
            self.tensors[name] = tensor
            spans.append((tensor.start, tensor.start + tensor.nbytes))
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("overlapping GGUF tensors")


class GGUFModelSource:
    def __init__(self, paths):
        """所有分片齐备且目录、架构一致才允许作为完整模型来源。"""
        shards = [GGUFShard(p) for p in paths]
        if not shards:
            raise ValueError("empty GGUF model source")
        # GGUF 后续片允许只携带 split 元数据，模型元数据只在第零片。
        shards.sort(key=lambda s: s.metadata.get("split.no", 0))
        self.metadata = shards[0].metadata
        if self.metadata.get("general.architecture") != "qwen4exp":
            raise ValueError("GGUF primary shard is not qwen4exp")
        count = self.metadata.get("split.count", 1)
        if len(shards) != count or {s.metadata.get("split.no", 0) for s in shards} != set(range(count)):
            raise ValueError("incomplete or duplicate GGUF split set")
        self.tensors = {}
        geometry = {k: v for k, v in self.metadata.items() if k.startswith("qwen4exp.")}
        for shard in shards:
            if shard.metadata.get("general.architecture", "qwen4exp") != "qwen4exp":
                raise ValueError("GGUF source is not qwen4exp")
            if shard.metadata.get("split.count", 1) != count:
                raise ValueError("GGUF split count mismatch")
            for key, value in shard.metadata.items():
                if key.startswith("qwen4exp.") and geometry.get(key) != value:
                    raise ValueError(f"GGUF split geometry mismatch: {key}")
            if shard.metadata.get("split.tensors.count", len(self.tensors)) != self.metadata.get("split.tensors.count", len(self.tensors)):
                raise ValueError("GGUF declared split tensor count mismatch")
            for name, tensor in shard.tensors.items():
                if name in self.tensors:
                    raise ValueError(f"duplicate split tensor {name}")
                self.tensors[name] = tensor
        expected = self.metadata.get("split.tensors.count", len(self.tensors))
        if len(self.tensors) != expected:
            raise ValueError(f"GGUF tensor count mismatch: {len(self.tensors)} != {expected}")


@dataclass(frozen=True)
class TensorMapping:
    target: str
    semantics: str = "identity"
    # 保留不同量化类型的独立段，由线性层按 output offset 拼接，禁止拼原始字节。
    segment: str | None = None
    head_layout: str = "native"


_GLOBAL = {
    "token_embd.weight": "embed_tokens.weight", "output.weight": "lm_head.weight",
    "output_hc_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
    "output_hc_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight",
    "output_hc_norm.weight": "hyper_connection_mixer.hc_norm.weight",
}
_LAYER = {
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_up_proj.weight:gate",
    "ffn_up_shexp.weight": "mlp.shared_expert.gate_up_proj.weight:up",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkvz.weight:qkv",
    "attn_gate.weight": "linear_attn.in_proj_qkvz.weight:z",
    "ssm_alpha.weight": "linear_attn.in_proj_ba.weight:a",
    "ssm_beta.weight": "linear_attn.in_proj_ba.weight:b",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log", "ssm_norm.weight": "linear_attn.norm.weight",
    "attn_q.weight": "self_attn.qkv_proj.weight:q",
    "attn_k.weight": "self_attn.qkv_proj.weight:k",
    "attn_v.weight": "self_attn.qkv_proj.weight:v",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "indexer.q_proj.weight": "self_attn.indexer.index_qk_proj.weight:q",
    "indexer.k_proj.weight": "self_attn.indexer.index_qk_proj.weight:k",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "ple_key.weight": "ple.key_proj.weight", "ple_value.weight": "ple.value_proj.weight",
    "ple_conv1d.weight": "ple.conv1d.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
}
for _kind, _module in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
    for _role, _target in (("down", "input_mix_weight_down_block_inject.weight:down"),
                           ("inject", "input_mix_weight_down_block_inject.weight:inject"),
                           ("up", "input_mix_weight_up.weight"), ("norm", "hc_norm.weight")):
        _LAYER[f"hc_{_kind}_{_role}.weight"] = f"{_module}.{_target}"


def map_tensor(name):
    """返回运行时目标和语义变换；未声明张量必须报错，不能静默丢弃。"""
    if name == "per_layer_token_embd.weight":
        return TensorMapping("ple_row_table", "iq4_nl_row_table")
    if name in _GLOBAL:
        target = _GLOBAL[name]
    else:
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if match is None:
            raise ValueError(f"unmapped GGUF tensor {name}")
        layer, suffix = match.groups()
        expert = re.fullmatch(r"ffn_(gate|up|down)_exps\.weight", suffix)
        if expert:
            return TensorMapping(f"layers.{layer}.mlp.experts.{expert[1]}", "expert_bank")
        if suffix not in _LAYER:
            raise ValueError(f"unmapped GGUF tensor {name}")
        target = f"layers.{layer}.{_LAYER[suffix]}"
    target, _, segment = target.partition(":")
    semantics = "identity"
    if target.endswith("A_log"):
        semantics = "negative_exp_A"
    elif target.endswith(("hc_norm.weight", "q_norm.weight", "k_norm.weight",
                          "q_layernorm.weight", "k_layernorm.weight",
                          "norm_key.weight", "norm_query.weight", "norm_conv.weight")):
        semantics = "folded_plus_one_norm"
    elif target.endswith("conv1d.weight"):
        semantics = "depthwise_conv"
    layout = "ggml_cyclic_gdn" if ".linear_attn." in target else "native"
    return TensorMapping(target, semantics, segment or None, layout)
