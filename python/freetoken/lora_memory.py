"""Torch-free resident-byte estimate for the FP32 fused adapter layout."""

import json
from pathlib import Path
import struct


def _shapes(path):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    if any(value["dtype"] != "F32" for name, value in header.items() if name != "__metadata__"):
        raise ValueError("The FP32 adapter layout requires F32 safetensors")
    return {name.removeprefix("base_model.model.").replace(".default.weight", ".weight"): value["shape"]
            for name, value in header.items() if name != "__metadata__"}


def adapter_resident_bytes(directory, model_config):
    """Read only safetensors headers; include block-diagonal packing overhead."""
    directory = Path(directory)
    config = model_config.get("text_config", model_config)
    shapes = _shapes(directory / "adapter_model.safetensors")
    shapes.update(_shapes(directory / "expert_lora.safetensors"))
    groups, experts = {}, {}
    for name, a in shapes.items():
        if name.endswith(".lora_A.weight"):
            target = name.removesuffix(".lora_A.weight")
            b = shapes[target + ".lora_B.weight"]
            group, width = target, b[0]
            if ".linear_attn.in_proj_" in target:
                group = target.rsplit(".", 1)[0] + ".in_proj_qkvz"
                width = (2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
                         + 2 * config["linear_num_value_heads"] * config["linear_value_head_dim"])
            elif target.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
                group = target.rsplit(".", 1)[0] + ".qkv_proj"
                width = 2 * (config["num_attention_heads"] + config["num_key_value_heads"]) * config["head_dim"]
            elif target.endswith((".shared_expert.gate_proj", ".shared_expert.up_proj")):
                group = target.rsplit(".", 1)[0] + ".gate_up_proj"
                width = 2 * config["shared_expert_intermediate_size"]
            entry = groups.setdefault(group, [a[1], width, 0])
            entry[2] += a[0]
        elif name.endswith(".lora_A"):
            target = name.removesuffix(".lora_A")
            b = shapes[target + ".lora_B"]
            group = target.rsplit(".", 1)[0]
            experts.setdefault(group, {})[target.rsplit(".", 1)[1]] = (a, b)
    total = sum((inp + out) * rank * 4 for inp, out, rank in groups.values())
    for projections in experts.values():
        gate_a, gate_b = projections["gate_proj"]
        up_a, up_b = projections["up_proj"]
        down_a, down_b = projections["down_proj"]
        e = gate_a[0]
        total += e * ((gate_a[1] + up_a[1]) * gate_a[2]
                      + (gate_b[1] + up_b[1]) * (gate_a[1] + up_a[1])
                      + down_a[1] * down_a[2] + down_b[1] * down_b[2]) * 4
    return total
