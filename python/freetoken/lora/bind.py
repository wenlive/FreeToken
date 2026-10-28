"""Map canonical HF adapter targets to FreeToken's fused projections."""

from __future__ import annotations

import torch

from .manifest import AdapterBundle


def validate_runtime(config) -> None:
    if (config.tp_info.size != 1 or config.use_dummy_weight or config.moe_strategy != "offload"
            or config.moe_cpu_layers or config.mtp_runtime_path is not None
            or config.moe_q2_resident_experts or not config.mm.text_model_only):
        raise ValueError("LoRA currently requires single-GPU text-only offload without CPU experts or MTP")


def bind_adapter(model, bundle: AdapterBundle, device) -> dict:
    from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    if not isinstance(model, Qwen3_5ForCausalLM):
        raise ValueError("This adapter loader supports Qwen3.5/3.6 text models only")
    if getattr(model, "_lora_identity", None) is not None:
        raise ValueError("An engine's adapter is immutable; create a new engine to change it")
    layers = model.model.layers.op_list
    destinations = {}

    def add(name, op, start, end):
        destinations[name] = (op, start, end)

    for index, layer in enumerate(layers):
        prefix = f"model.layers.{index}"
        if layer._is_linear:
            attn = layer.linear_attn
            op = attn.in_proj_qkvz if attn._split_in_proj else attn.in_proj
            add(f"{prefix}.linear_attn.in_proj_qkv", op, 0, attn.conv_dim)
            add(f"{prefix}.linear_attn.in_proj_z", op, attn.conv_dim, attn.conv_dim + attn.value_dim)
            add(f"{prefix}.linear_attn.out_proj", attn.out_proj, 0, attn.out_proj.out_features)
        else:
            attn = layer.self_attn
            start = 0
            for name, width in zip(("q_proj", "k_proj", "v_proj"), attn._qkv_split):
                add(f"{prefix}.self_attn.{name}", attn.qkv_proj, start, start + width)
                start += width
            add(f"{prefix}.self_attn.o_proj", attn.o_proj, 0, attn.o_proj.out_features)
        shared = layer.mlp.shared_expert
        width = shared.gate_up_proj.out_features // 2
        add(f"{prefix}.mlp.shared_expert.gate_proj", shared.gate_up_proj, 0, width)
        add(f"{prefix}.mlp.shared_expert.up_proj", shared.gate_up_proj, width, 2 * width)
        add(f"{prefix}.mlp.shared_expert.down_proj", shared.down_proj, 0, shared.down_proj.out_features)

    # Validate every destination before allocating or mutating model operators.
    for name, pair in bundle.linear.items():
        if name not in destinations:
            raise ValueError(f"Unknown model LoRA target: {name}")
        op, start, end = destinations[name]
        if pair.a.shape[1] != op.in_features or pair.b.shape[0] != end - start:
            raise ValueError(f"LoRA shape does not match the fused projection: {name}")
    for index, pairs in bundle.experts.items():
        if index >= len(layers):
            raise ValueError(f"Unknown expert layer: {index}")
        op = layers[index].mlp.experts
        if (op.quant_method is None or not isinstance(op.quant_method.kernel, TritonNvfp4MoEKernel)
                or op.apply_router_weight_on_input):
            raise ValueError("Expert LoRA requires the Triton NVFP4 kernel with output router weights")
        e, h, i = op.num_experts, op.hidden_size, op.intermediate_size
        for name, pair in pairs.items():
            inp, out = (i, h) if name == "down" else (h, i)
            if pair.a.shape != (e, pair.rank, inp) or pair.b.shape != (e, out, pair.rank):
                raise ValueError(f"Expert LoRA shape mismatch: layer {index} {name}")

    torch.backends.cuda.matmul.allow_tf32 = False
    for name, pair in bundle.linear.items():
        op, start, end = destinations[name]
        parts = getattr(op, "_lora_parts", ())
        op._lora_parts = (*parts, (start, end, pair.to(device)))
    for index, pairs in bundle.experts.items():
        layers[index].mlp.experts._lora_experts = {name: pair.to(device) for name, pair in pairs.items()}
    model._lora_identity = bundle.identity
    return {"identity": bundle.identity, "path": str(bundle.directory), "linear_projections": len(bundle.linear),
            "expert_layers": len(bundle.experts), "parameter_count": bundle.parameter_count,
            "parameter_bytes": bundle.parameter_bytes}
