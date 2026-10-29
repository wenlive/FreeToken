"""Dense adapters bind to the original deployment checkpoint, never a merged base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch
from safetensors.torch import load_file

from .manifest import AdapterBundle, _canonical, _file, _pair, _read, _scaling


_TARGET = re.compile(
    r"^(model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)))\.lora_([AB])\.weight$"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dense_adapter(directory: str | Path, checkpoint: str | Path) -> AdapterBundle:
    """Validate the dense contract and all declared base bytes once at engine startup."""
    directory, checkpoint = Path(directory).resolve(), Path(checkpoint).resolve()
    manifest = _read(_file(directory, "manifest.json"))
    if (manifest.get("layout_version") != "nvfp4-lora-dense-v1"
            or manifest.get("family") != "qwen3_5_dense" or manifest.get("status") != "complete"
            or manifest.get("synthetic_vocabulary_only") is not False
            or manifest.get("enable_thinking") is not False or manifest.get("linear_format") != "peft-lora"):
        raise ValueError("Unsupported, incomplete or synthetic dense adapter")
    base = manifest["base"]
    fingerprint = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = base["source"]
    if (base.get("family") != "qwen3_5_dense" or fingerprint != manifest["base_fingerprint"]
            or source.get("kind") != "real-dense-checkpoint"
            or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", ""))
            or len(source["shards"]) != 3 or not source.get("metadata")):
        raise ValueError("Dense base identity or revision is invalid")
    files = manifest["files"]
    if set(files) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("Dense adapter requires exactly its two declared PEFT files")
    for name, digest in files.items():
        if _sha(_file(directory, name)) != digest:
            raise ValueError(f"Dense adapter checksum mismatch: {name}")
    info = manifest["nonexpert"]
    scale = _scaling(info, expert=False)
    peft = _read(directory / "adapter_config.json")
    targets = info["target_modules"]
    if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
            or peft.get("peft_type") != "LORA" or peft.get("task_type") != "CAUSAL_LM"
            or peft.get("base_model_name_or_path") != source["repo"]
            or peft.get("r") != info["rank"] or peft.get("lora_alpha") != info["alpha"]
            or peft.get("target_modules") != targets or peft.get("use_rslora", False)
            or peft.get("use_dora", False) or peft.get("fan_in_fan_out", False)
            or peft.get("bias", "none") != "none" or peft.get("lora_dropout", 0) != 0
            or peft.get("modules_to_save")):
        raise ValueError("Dense PEFT configuration differs from manifest")
    specs = manifest["tensor_specifications"]
    tensors = load_file(directory / "adapter_model.safetensors", device="cpu")
    values, observed = {}, set()
    count = size = 0
    for key, value in tensors.items():
        name = _canonical(key)
        match = _TARGET.fullmatch(name)
        spec = specs.get(name)
        if (not key.startswith("base_model.model.model.layers.") or match is None or spec is None
                or name in observed or spec.get("file") != "adapter_model.safetensors" or spec.get("key") != key
                or spec.get("dtype") != "float32" or value.dtype != torch.float32
                or list(value.shape) != spec.get("shape") or value.numel() != spec.get("numel")
                or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid dense adapter tensor: {key}")
        observed.add(name)
        values.setdefault(match[1], {})[match[2]] = value
        count += value.numel()
        size += value.numel() * value.element_size()
    if observed != set(specs) or set(values) != set(targets):
        raise ValueError("Dense declared targets and tensor set differ")
    linear = {name: _pair(pair, info["rank"], 2, scale) for name, pair in values.items()}
    declared_base = {}
    for item in [*source["metadata"], *source["shards"]]:
        name, digest = item["filename"], item["sha256"]
        if name in declared_base or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Duplicate or invalid dense base file declaration")
        declared_base[name] = digest
        if _sha(_file(checkpoint, name)) != digest:
            raise ValueError(f"Dense deployment checkpoint checksum mismatch: {name}")
    for name, digest in base["tokenizer_source_hashes"].items():
        if declared_base.get(name) != digest:
            raise ValueError(f"Dense tokenizer identity differs: {name}")
    identity = hashlib.sha256((fingerprint + files["adapter_model.safetensors"]
                               + files["adapter_config.json"]).encode()).hexdigest()
    return AdapterBundle(directory, identity, manifest, linear, {}, count, size)
