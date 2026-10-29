"""Bind dense HF projections to the existing fused inference operators."""

from __future__ import annotations

import torch

from .ops import pack_linear


def bind_dense_adapter(model, bundle, device) -> dict:
    from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM

    if not isinstance(model, Qwen3_5ForCausalLM):
        raise ValueError("Dense LoRA requires a Qwen3.5/3.6/3.8 text model")
    if getattr(model, "_lora_identity", None) is not None:
        raise ValueError("An engine's adapter is immutable; create a new engine to change it")
    if bundle.experts or bundle.manifest.get("layout_version") != "nvfp4-lora-dense-v1":
        raise ValueError("Dense binding requires the dense adapter layout")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("FP32 LoRA requires IEEE matmul; disable TF32 before constructing the engine")
    destinations = {}
    for index, layer in enumerate(model.model.layers.op_list):
        prefix = f"model.layers.{index}"
        if hasattr(layer.mlp, "shared_expert") or hasattr(layer.mlp, "experts"):
            raise ValueError("Dense adapters cannot bind to MoE layers")
        if layer._is_linear:
            attn = layer.linear_attn
            op = attn.in_proj_qkvz if attn._split_in_proj else attn.in_proj
            destinations[f"{prefix}.linear_attn.in_proj_qkv"] = (op, 0, attn.conv_dim)
            destinations[f"{prefix}.linear_attn.in_proj_z"] = (op, attn.conv_dim, attn.conv_dim + attn.value_dim)
            destinations[f"{prefix}.linear_attn.out_proj"] = (attn.out_proj, 0, attn.out_proj.out_features)
        else:
            attn, start = layer.self_attn, 0
            for name, width in zip(("q_proj", "k_proj", "v_proj"), attn._qkv_split):
                destinations[f"{prefix}.self_attn.{name}"] = (attn.qkv_proj, start, start + width)
                start += width
            destinations[f"{prefix}.self_attn.o_proj"] = (attn.o_proj, 0, attn.o_proj.out_features)
        width = layer.mlp.gate_up_proj.out_features // 2
        destinations[f"{prefix}.mlp.gate_proj"] = (layer.mlp.gate_up_proj, 0, width)
        destinations[f"{prefix}.mlp.up_proj"] = (layer.mlp.gate_up_proj, width, 2 * width)
        destinations[f"{prefix}.mlp.down_proj"] = (layer.mlp.down_proj, 0, layer.mlp.down_proj.out_features)
    grouped = {}
    for name, pair in bundle.linear.items():
        if name not in destinations:
            raise ValueError(f"Unknown dense LoRA target: {name}")
        op, start, end = destinations[name]
        if pair.a.shape[1] != op.in_features or pair.b.shape[0] != end - start:
            raise ValueError(f"Dense LoRA fused projection shape mismatch: {name}")
        grouped.setdefault(id(op), (op, []))[1].append((start, end, pair))
    # Prepare all allocations first so a failure cannot leave a partly bound model.
    packed = [(op, pack_linear(parts, op.in_features, op.out_features, device)) for op, parts in grouped.values()]
    resident = sum(t.numel() * t.element_size() for _, item in packed for t in (item.a, item.b))
    for op, item in packed:
        op._lora_linear = item
    model._lora_identity = bundle.identity
    return {"identity": bundle.identity, "path": str(bundle.directory), "linear_projections": len(bundle.linear),
            "expert_layers": 0, "parameter_count": bundle.parameter_count, "parameter_bytes": bundle.parameter_bytes,
            "resident_bytes": resident, "packed_linear_operations": len(packed)}
