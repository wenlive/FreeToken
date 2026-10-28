"""Read pickle-free training artifacts and bind them to the original checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from safetensors.torch import load_file


_LINEAR = re.compile(
    r"^(model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)))\.lora_([AB])\.weight$"
)
_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj\.lora_([AB])$")


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Artifact file names must be relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing file or file outside its root: {name}")
    return path


def _canonical(name: str) -> str:
    return name.removeprefix("base_model.model.").replace(
        ".lora_A.default.weight", ".lora_A.weight"
    ).replace(".lora_B.default.weight", ".lora_B.weight")


def _scaling(info: dict, *, expert: bool):
    rank, alpha = info["rank"], info["alpha"]
    if (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
            or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0
            or info["rsLoRA"] is not False or info["dtype"] != "float32"):
        raise ValueError("Only finite, positive-rank ordinary FP32 LoRA is supported")
    scales = info["scaling"] if expert else [info["scaling"]]
    if not isinstance(scales, list) or len(scales) != (3 if expert else 1):
        raise ValueError("Unexpected LoRA scaling layout")
    if not all(isinstance(scale, (int, float)) and math.isfinite(scale)
               and math.isclose(scale, alpha / rank, rel_tol=0, abs_tol=1e-12) for scale in scales):
        raise ValueError("LoRA scaling does not match alpha/r")
    return tuple(float(scale) for scale in scales) if expert else float(scales[0])


@dataclass(frozen=True)
class LoRAPair:
    a: torch.Tensor
    b: torch.Tensor
    scaling: float

    @property
    def rank(self) -> int:
        return self.a.shape[-2]

    def to(self, device) -> "LoRAPair":
        return LoRAPair(self.a.to(device).contiguous(), self.b.to(device).contiguous(), self.scaling)


@dataclass(frozen=True)
class AdapterBundle:
    directory: Path
    identity: str
    manifest: dict
    linear: dict[str, LoRAPair]
    experts: dict[int, dict[str, LoRAPair]]
    parameter_count: int
    parameter_bytes: int


def _pair(values: dict[str, torch.Tensor], rank: int, dimensions: int, scaling: float) -> LoRAPair:
    if set(values) != {"A", "B"}:
        raise ValueError("Each LoRA projection needs exactly one A and one B")
    a, b = values["A"], values["B"]
    if (a.ndim != dimensions or b.ndim != dimensions or a.shape[-2] != rank or b.shape[-1] != rank
            or (dimensions == 3 and a.shape[0] != b.shape[0]) or min(*a.shape, *b.shape) <= 0):
        raise ValueError("LoRA A/B shapes or rank do not match")
    return LoRAPair(a, b, scaling)


def _verify_base(manifest: dict, checkpoint: Path) -> None:
    base = manifest["base"]
    fingerprint = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != manifest["base_fingerprint"]:
        raise ValueError("Adapter base fingerprint is inconsistent")
    source = base["source"]
    if source["kind"] != "real-nvidia-checkpoint":
        raise ValueError("This loader requires a real NVIDIA NVFP4 training base")
    shards = source["shards"]
    if not shards or not source["metadata"]:
        raise ValueError("Base identity lacks checkpoint files")
    files = [*source["metadata"], *shards]
    if len({item["filename"] for item in files}) != len(files):
        raise ValueError("Duplicate base identity file")
    for item in files:
        if _hash(_file(checkpoint, item["filename"])) != item["sha256"]:
            raise ValueError(f"Base checksum mismatch: {item['filename']}")
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file() and set(_read(index)["weight_map"].values()) != {item["filename"] for item in shards}:
        raise ValueError("Base shard index differs from the adapter source identity")


def read_adapter(directory: str | Path, checkpoint: str | Path) -> AdapterBundle:
    """Validate bytes and canonical names before returning CPU FP32 pairs.

    The training bundle's BF16 derived files are not inference weights. Its source
    identity binds this artifact to the original packed checkpoint used by FreeToken.
    """
    directory, checkpoint = Path(directory).resolve(), Path(checkpoint).resolve()
    manifest_path = _file(directory, "manifest.json")
    manifest = _read(manifest_path)
    if (manifest["layout_version"] != "nvfp4-lora-v1" or manifest["status"] != "complete"
            or manifest["synthetic_vocabulary_only"] is not False or manifest["enable_thinking"] is not False):
        raise ValueError("Unsupported, incomplete or synthetic adapter")
    required = {"adapter_config.json", "adapter_model.safetensors", "expert_lora.safetensors"}
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete adapter file inventory")
    for filename, expected in manifest["files"].items():
        if _hash(_file(directory, filename)) != expected:
            raise ValueError(f"Adapter checksum mismatch: {filename}")
    _verify_base(manifest, checkpoint)

    info = manifest["nonexpert"]
    linear_scale = _scaling(info, expert=False) if info is not None else None
    peft = _read(directory / "adapter_config.json")
    if info is not None:
        if (manifest["linear_format"] != "peft-lora" or peft["peft_type"] != "LORA"
                or peft["r"] != info["rank"] or peft["lora_alpha"] != info["alpha"]
                or peft.get("use_rslora", False) or peft.get("use_dora", False)
                or peft.get("fan_in_fan_out", False) or peft.get("bias", "none") != "none"
                or peft.get("lora_dropout", 0) != 0 or peft["target_modules"] != info["target_modules"]):
            raise ValueError("PEFT configuration differs from the supported nonexpert layout")
    elif manifest["linear_format"] != "none":
        raise ValueError("Nonexpert layout is inconsistent")
    if manifest["expert_layout"] != "{gate,up,down}_proj.lora_A[E,r,in] and lora_B[E,out,r]":
        raise ValueError("Unsupported expert LoRA layout")
    expert_info = {int(layer): item for layer, item in manifest["experts"].items()}
    if any(str(int(layer)) != layer or int(layer) < 0 for layer in manifest["experts"]):
        raise ValueError("Expert layer names must be canonical nonnegative integers")
    expert_scales = {layer: _scaling(item, expert=True) for layer, item in expert_info.items()}
    specs = manifest["tensor_specifications"]
    if not specs:
        raise ValueError("Adapter has no trainable tensors")
    declared = {}
    for canonical, spec in specs.items():
        filename, key = spec["file"], spec["key"]
        if filename not in {"adapter_model.safetensors", "expert_lora.safetensors"} or _canonical(key) != canonical:
            raise ValueError(f"Invalid canonical tensor mapping: {canonical}")
        if (filename, key) in declared:
            raise ValueError("Two canonical names refer to the same tensor")
        declared[filename, key] = canonical
    linear_values, expert_values = {}, {}
    parameter_count = parameter_bytes = 0
    observed = set()
    for filename in ["adapter_model.safetensors", "expert_lora.safetensors"]:
        for key, value in load_file(directory / filename, device="cpu").items():
            canonical = declared.get((filename, key))
            if canonical is None:
                raise ValueError(f"Undeclared adapter tensor: {key}")
            observed.add((filename, key))
            spec = specs[canonical]
            if (value.dtype != torch.float32 or spec["dtype"] != "float32" or list(value.shape) != spec["shape"]
                    or value.numel() != spec["numel"] or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid adapter tensor shape, dtype or values: {canonical}")
            parameter_count += value.numel()
            parameter_bytes += value.numel() * value.element_size()
            linear = _LINEAR.fullmatch(canonical)
            expert = _EXPERT.fullmatch(canonical)
            if linear is not None and filename == "adapter_model.safetensors" and info is not None:
                linear_values.setdefault(linear[1], {})[linear[2]] = value
            elif expert is not None and filename == "expert_lora.safetensors":
                layer, projection, side = int(expert[1]), expert[2], expert[3]
                expert_values.setdefault(layer, {}).setdefault(projection, {})[side] = value
            else:
                raise ValueError(f"Unsupported adapter target: {canonical}")
    if observed != set(declared):
        raise ValueError("Declared and actual adapter tensor sets differ")
    if bool(linear_values) != (info is not None) or set(expert_values) != set(expert_info):
        raise ValueError("Adapter parameter groups differ from manifest metadata")
    linear = {name: _pair(values, info["rank"], 2, linear_scale) for name, values in linear_values.items()}
    experts = {}
    for layer, values in expert_values.items():
        if set(values) != {"gate", "up", "down"}:
            raise ValueError("Expert adapters require gate, up and down projections")
        experts[layer] = {
            projection: _pair(values[projection], expert_info[layer]["rank"], 3, expert_scales[layer][index])
            for index, projection in enumerate(("gate", "up", "down"))
        }
    return AdapterBundle(directory, _hash(manifest_path), manifest, linear, experts, parameter_count, parameter_bytes)
