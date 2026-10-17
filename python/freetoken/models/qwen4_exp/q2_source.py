"""Expert-only GGUF v3 source; deliberately independent of GGUF trunk/PLE readers."""
from __future__ import annotations

import math
from pathlib import Path
import re

import torch

from freetoken.checkpoint.gguf_reader import (
    read_exact as _read,
    unpack as _unpack,
    read_string as _string,
    read_value as _value,
)


GGML_GSQ_Q2_0 = 42
Q2_BLOCK_ELEMENTS = 64
Q2_BLOCK_BYTES = 18
def read_q2_directory(path: Path, *, layers, experts, hidden, intermediate):
    """按调用方模型几何校验完整专家目录，不把某个发布文件当作格式定义。"""
    if (layers <= 0 or experts <= 0 or hidden <= 0 or intermediate <= 0
            or hidden % Q2_BLOCK_ELEMENTS or intermediate % Q2_BLOCK_ELEMENTS):
        raise ValueError("invalid GSQ Q2_0 model geometry")
    metadata, directory = {}, {}
    with path.open("rb") as stream:
        magic, version, tensors, values = _unpack(stream, "IIQQ")
        if magic != 0x46554747 or version != 3 or tensors > 100_000 or values > 100_000:
            raise ValueError("invalid GGUF v3 header")
        for _ in range(values):
            key = _string(stream)
            value = _value(stream, keep=not key.startswith("tokenizer."))
            if not key.startswith("tokenizer."):
                metadata[key] = value
        for _ in range(tensors):
            name = _string(stream)
            dims, = _unpack(stream, "I")
            if not 1 <= dims <= 4:
                raise ValueError("invalid GGUF tensor rank")
            shape = _unpack(stream, "Q" * dims)
            kind, offset = _unpack(stream, "IQ")
            if name in directory:
                raise ValueError(f"duplicate GGUF tensor {name}")
            directory[name] = (shape, kind, offset)
        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("invalid GGUF alignment")
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
    required = {"general.architecture": "qwen4exp", "qwen4exp.block_count": layers,
                "qwen4exp.expert_count": experts, "qwen4exp.embedding_length": hidden,
                "qwen4exp.expert_feed_forward_length": intermediate}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Q2_0 geometry mismatch: {key}={metadata.get(key)!r}, expected {expected!r}")
    selected, spans = {}, []
    source_size = path.stat().st_size
    for layer in range(layers):
        for role in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            expected = (intermediate, hidden, experts) if role == "down" else (hidden, intermediate, experts)
            shape, kind, offset = directory.get(name, (None, None, None))
            if shape != expected or kind != GGML_GSQ_Q2_0:
                raise ValueError(f"missing or incompatible Q2_0 expert tensor {name}: {shape}, type {kind}")
            size = math.prod(shape) // Q2_BLOCK_ELEMENTS * Q2_BLOCK_BYTES
            start = data_start + offset
            if start < data_start or start + size > source_size or offset % alignment:
                raise ValueError(f"out-of-bounds or misaligned expert tensor {name}")
            selected[layer, role] = (start, size)
            spans.append((start, start + size))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping GGUF expert tensors")
    actual = {name for name in directory if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", name)}
    if len(actual) != layers * 3:
        raise ValueError("unexpected routed-expert tensor count")
    return selected


def iter_q2_expert_pieces(model_path, config, *, chunk=8 << 20):
    """校验已验字节的来源身份，再分块读取全部专家；不在热启动重复扫描大文件。"""
    source = config.expert_source
    if (not isinstance(source, dict) or source.get("format") != "q2_0"
            or source.get("scope") != "routed_experts_only"):
        raise ValueError("Q2_0 requires an explicit expert source")
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    geometry = {"layers": config.num_layers, "experts_per_layer": E,
                "hidden": H, "intermediate": I,
                "expert_bytes": 3 * config.num_layers * E * H * I // Q2_BLOCK_ELEMENTS * Q2_BLOCK_BYTES}
    if any(source.get(key) != value for key, value in geometry.items()):
        raise ValueError("Q2_0 source contract does not match model geometry")
    digest = source.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Q2_0 source needs a verified SHA256 receipt")
    path = Path(source["path"])
    if not path.is_absolute():
        path = Path(model_path) / path
    stat = path.stat()
    if (stat.st_size != source.get("size")
            or source.get("verified_mtime_ns") != stat.st_mtime_ns):
        raise ValueError("Q2_0 source identity changed; reverify the complete artifact")
    directory = read_q2_directory(path, layers=config.num_layers, experts=config.num_experts,
                                  hidden=config.hidden_size, intermediate=config.moe_intermediate_size)
    per_expert = 3 * H * I // 64 * 18
    count = max(1, min(E, chunk // per_expert))
    # 按专家块读取，保持所有层和专家完整，禁止缺失专家别名。
    with path.open("rb", buffering=0) as stream:
        for layer in range(config.num_layers):
            for e0 in range(0, E, count):
                e1 = min(E, e0 + count)
                pieces = {}
                for role in ("gate", "up", "down"):
                    rows, width = (H, I // 64 * 18) if role == "down" else (I, H // 64 * 18)
                    piece = torch.empty((e1 - e0, rows, width), dtype=torch.uint8)
                    dst = memoryview(piece.numpy()).cast("B")
                    start, total = directory[layer, role]
                    stream.seek(start + e0 * (total // E))
                    done = 0
                    while done < len(dst):
                        got = stream.readinto(dst[done:])
                        if not got:
                            raise ValueError("short Q2_0 expert read")
                        done += got
                    pieces[role] = piece
                yield layer, e0, e1, pieces
