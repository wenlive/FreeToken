"""Native Clover configuration; causal geometry is shared with Qwen3.5/3.6."""
from dataclasses import replace

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.models.qwen3_5_moe.config import parse_config as parse_qwen


def parse_config(hf_config):
    if getattr(hf_config, "model_type", None) != "clover_moe":
        raise ValueError("Clover requires model_type=clover_moe")
    config = parse_qwen(hf_config)
    # Qwen's expert probe is anchored to its multimodal wrapper. Clover keeps
    # native model.layers names; do not misclassify its expert banks as BF16.
    scheme = QuantConfig.from_hf(hf_config).scheme_for_name("model.layers.0.mlp.experts.0.gate_proj")
    if scheme is not None and scheme.kind is not QuantKind.NVFP4:
        raise NotImplementedError("Clover currently supports BF16 or NVFP4 routed experts")
    return replace(config, expert_quant="none" if scheme is None else str(scheme.kind), weight_block_size=None)
