"""Stream Clover's routed experts into ModelOpt W4A16 NVFP4.

The source checkpoint stays read-only. Output uses native Clover names and the
existing FreeToken NVFP4 bank contract: E2M1, adjacent low/high nibbles,
E4M3 group-16 scales, and a dequantization-side global scale per projection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import time

import safetensors
from safetensors.torch import save_file
import torch

_STACKED = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
_AUXILIARY = ("bd_gdn_conv_right_weight.safetensors", "bd_gdn_softmax_scalars.safetensors")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Round to nearest E2M1, ties to even significand, preserving signed zero."""
    boundaries = values.new_tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    magnitude = values.abs().contiguous()
    code = torch.bucketize(magnitude, boundaries)
    tie = magnitude == boundaries[code.clamp_max(6)]
    code += (tie & (code.remainder(2) == 1)).to(code.dtype)
    return (code | (torch.signbit(values).to(code.dtype) << 3)).to(torch.uint8)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2 or codes.dtype != torch.uint8:
        raise ValueError("E2M1 packing requires uint8 codes and an even last dimension")
    if bool((codes > 15).any()):
        raise ValueError("E2M1 code is outside [0,15]")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def dequantize(weight, weight_scale, weight_scale_2):
    """Portable reference decoder, also usable by the HF oracle loader."""
    lut = torch.tensor((*E2M1, *(-x for x in E2M1)), device=weight.device)
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
    values = lut[codes]
    scales = weight_scale.float().repeat_interleave(16, dim=-1)
    return values * scales * weight_scale_2.float()


def quantize(weight: torch.Tensor):
    if weight.ndim != 2 or weight.shape[-1] % 16 or not weight.numel():
        raise ValueError("NVFP4 requires a nonempty matrix with input width divisible by 16")
    values = weight.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Cannot quantize non-finite weights")
    # The existing inference bank keeps globals in FP16. Store that exact value
    # in the FP32 scalar too, so the HF dequantization oracle sees the same scale.
    global_scale = (values.abs().amax() / (448.0 * 6.0)).clamp_min(2.0 ** -24)
    global_scale = global_scale.to(torch.float16).float()
    if not bool(torch.isfinite(global_scale)):
        raise ValueError("NVFP4 global scale is not representable by the inference bank")
    groups = values.reshape(*values.shape[:-1], -1, 16)
    scales = (groups.abs().amax(-1) / (6.0 * global_scale)).clamp(0.0, 448.0)
    scales = scales.to(torch.float8_e4m3fn)
    divisor = scales.float() * global_scale
    normalized = groups / divisor.clamp_min(torch.finfo(torch.float32).tiny)[..., None]
    codes = encode_e2m1(normalized).reshape_as(values)
    return pack_e2m1(codes), scales, global_scale.reshape(())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_shard(destination, tensors, weight_map, receipts):
    temporary = destination.with_suffix(destination.suffix + ".partial")
    save_file(tensors, str(temporary), metadata={"format": "pt"})
    temporary.replace(destination)
    for name in tensors:
        if name in weight_map:
            raise ValueError(f"Duplicate output tensor: {name}")
        weight_map[name] = destination.name
    receipts[destination.name] = dict(bytes=destination.stat().st_size, sha256=sha256(destination))


def convert(source: Path, output: Path, *, device="cuda", sample_experts=3, seed=20261030):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the read-only source directory")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "clover_moe":
        raise ValueError("Expected a native Clover checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    source_map = index["weight_map"]
    layers, experts = int(config["num_hidden_layers"]), int(config["num_experts"])
    hidden, intermediate = int(config["hidden_size"]), int(config["moe_intermediate_size"])
    routed = {name for name in source_map if _STACKED.fullmatch(name)}
    if len(routed) != 2 * layers or not 1 <= sample_experts <= experts:
        raise ValueError("Incomplete routed-expert layout or invalid sample count")
    # Validate every indexed shard before creating any output; partial-download
    # files and unindexed safetensors never become accidental converter inputs.
    source_receipts = {}
    for filename in sorted(set(source_map.values())):
        path = source / filename
        if path.parent != source or not path.is_file():
            raise ValueError(f"Invalid source shard {filename!r}")
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            indexed_names = {n for n, f in source_map.items() if f == filename}
            if indexed_names != set(handle.keys()):
                raise ValueError(f"Shard/index disagreement in {filename}")
            for name in indexed_names & routed:
                shape = [experts, 2 * intermediate, hidden] if name.endswith("gate_up_proj") else [experts, hidden, intermediate]
                if handle.get_slice(name).get_shape() != shape:
                    raise ValueError(f"Unexpected stacked expert shape for {name}")
        source_receipts[filename] = dict(bytes=path.stat().st_size)
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(schema_version=1, state="running", source=str(source),
                    source_index_sha256=sha256(source / "model.safetensors.index.json"),
                    source_config_sha256=sha256(source / "config.json"),
                    recipe="routed-nvfp4-w4a16-group16-fp16-exact-global-v1",
                    seed=seed, sample_experts_per_layer=sample_experts,
                    source_files=source_receipts, output_files={}, reconstruction=[])
    _write_json(output / "conversion.json", manifest)
    weight_map, receipts = {}, manifest["output_files"]
    total_payload = 0
    rng = random.Random(seed)
    try:
        for layer in range(layers):
            tensors = {}
            sampled = set(rng.sample(range(experts), sample_experts))
            stack_names = {proj: f"model.layers.{layer}.mlp.experts.{proj}" for proj in ("gate_up_proj", "down_proj")}
            handles = {}
            try:
                for filename in {source_map[name] for name in stack_names.values()}:
                    handles[filename] = safetensors.safe_open(source / filename, framework="pt", device="cpu")
                    handles[filename].__enter__()
                for expert in range(experts):
                    for projection in ("gate", "up", "down"):
                        stacked_name = stack_names["down_proj" if projection == "down" else "gate_up_proj"]
                        view = handles[source_map[stacked_name]].get_slice(stacked_name)
                        if projection == "down":
                            original = view[expert, :, :]
                        else:
                            begin = 0 if projection == "gate" else intermediate
                            original = view[expert, begin:begin + intermediate, :]
                        weight = original.to(device)
                        packed, scale, glob = quantize(weight)
                        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
                        for kind, tensor in (("weight", packed), ("weight_scale", scale), ("weight_scale_2", glob)):
                            value = tensor.cpu().contiguous()
                            tensors[f"{prefix}.{kind}"] = value
                            total_payload += value.numel() * value.element_size()
                        if expert in sampled:
                            reference = weight.float()
                            error = (dequantize(packed, scale, glob) - reference).norm() / reference.norm().clamp_min(1e-30)
                            manifest["reconstruction"].append(dict(layer=layer, expert=expert, projection=projection, relative_l2=float(error)))
                        del weight, original, packed, scale, glob
            finally:
                for handle in handles.values():
                    handle.__exit__(None, None, None)
            filename = output / f"experts-{layer:02d}.safetensors"
            _save_shard(filename, tensors, weight_map, receipts)
            del tensors
            manifest.update(completed_layers=layer + 1, elapsed_seconds=time.monotonic() - started)
            _write_json(output / "conversion.json", manifest)
            print(json.dumps(dict(layer=layer, completed=layer + 1, total=layers, elapsed_seconds=manifest["elapsed_seconds"])), flush=True)
        by_file = defaultdict(list)
        for name, filename in source_map.items():
            if name not in routed:
                by_file[filename].append(name)
        for ordinal, (filename, names) in enumerate(sorted(by_file.items())):
            if filename in _AUXILIARY:
                shutil.copy2(source / filename, output / filename)
                receipts[filename] = dict(bytes=(output / filename).stat().st_size, sha256=sha256(output / filename))
                with safetensors.safe_open(output / filename, framework="pt", device="cpu") as handle:
                    for name in names:
                        tensor = handle.get_slice(name)
                        count = 1
                        for dim in tensor.get_shape(): count *= dim
                        total_payload += count * {"BF16": 2, "F32": 4}[tensor.get_dtype()]
                        weight_map[name] = filename
            else:
                with safetensors.safe_open(source / filename, framework="pt", device="cpu") as handle:
                    tensors = {name: handle.get_tensor(name) for name in names}
                    total_payload += sum(t.numel() * t.element_size() for t in tensors.values())
                    _save_shard(output / f"nonexpert-{ordinal:02d}.safetensors", tensors, weight_map, receipts)
                    del tensors
        for path in source.iterdir():
            if path.is_file() and (path.suffix in (".py", ".md", ".jinja") or path.name in ("LICENSE", "tokenizer.json", "tokenizer_config.json", "generation_config.json")):
                shutil.copy2(path, output / path.name)
        quant = dict(quant_method="modelopt", quant_algo="MIXED_PRECISION", quantized_layers={
            f"model.layers.{layer}.mlp.experts": dict(quant_algo="W4A16_NVFP4", group_size=16)
            for layer in range(layers)
        })
        config["quantization_config"] = quant
        _write_json(output / "config.json", config)
        _write_json(output / "hf_quant_config.json", {"quantization": quant})
        expected_tensors = len(source_map) - len(routed) + layers * experts * 9
        if len(weight_map) != expected_tensors:
            raise ValueError(f"Output completeness error: {len(weight_map)} != {expected_tensors}")
        _write_json(output / "model.safetensors.index.json", dict(metadata=dict(total_size=total_payload), weight_map=weight_map))
        errors = [m["relative_l2"] for m in manifest["reconstruction"]]
        manifest.update(state="complete", tensor_count=len(weight_map), payload_bytes=total_payload,
                        elapsed_seconds=time.monotonic() - started,
                        relative_l2=dict(min=min(errors), mean=sum(errors) / len(errors), max=max(errors)))
    except BaseException as exc:
        manifest.update(state="failed", error=repr(exc), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        _write_json(output / "conversion.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-experts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = convert(args.source, args.output, device=args.device, sample_experts=args.sample_experts, seed=args.seed)
    print(json.dumps({k: result[k] for k in ("state", "tensor_count", "payload_bytes", "relative_l2", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
