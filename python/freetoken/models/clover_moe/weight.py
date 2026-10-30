"""Clover dense tensors and per-expert ModelOpt banks under native key names."""
import re

from freetoken.layers.quantization import QuantKind, get_quant_config
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.models.loader import ShardReader
from freetoken.models.qwen3_5_moe.weight import _is_gemma_norm, iter_weights as _qwen_weights, iter_weights_parallel


def iter_weights(model_path, device, *, include_moe_experts, include_non_moe, include_vision=True):
    reader = ShardReader(model_path, device) if include_non_moe else None
    try:
        for name, tensor in _qwen_weights(model_path, device, include_moe_experts=include_moe_experts,
                                          include_non_moe=include_non_moe, include_vision=include_vision):
            # Qwen's loader folds (1+w) in BF16. Clover's official function adds
            # in FP32; use the original small delta tensor, never subtract 1 back.
            yield name, reader.get_tensor(name) if reader is not None and _is_gemma_norm(name) else tensor
    finally:
        if reader is not None:
            reader.close()


def nvfp4_expert_spec(model_path, config):
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kinds = {stored[role].name: canonical for role, canonical in (
        ("weight", "weight"), ("weight_scale", "weight_scale"), ("weight_global", "weight_scale_2")
    )}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(
            r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
            r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>" + "|".join(map(re.escape, kinds)) + r")$"
        ),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,
        desc="Clover native NVFP4 experts",
        kind_map=kinds,
        global_reciprocal=stored["weight_global"].reciprocal,
    )
