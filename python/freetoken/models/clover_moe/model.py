"""Clover causal model with explicit storage for the read-only preview weights.

Architecture reference: DiffuSpace/Clover-35B-A3B-Preview,
4b0402b65e523d0183a14c431955438714cb16ff/modeling_clover_moe.py.
No HF implementation is copied here; causal operators reuse FreeToken's Qwen
path, leaving ordinary Qwen and Flash-Next arithmetic unchanged.
"""
import torch

from freetoken.core import get_global_ctx
from freetoken.layers.norm import GemmaRMSNorm
from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM


class CloverPlusOneRMSNorm(GemmaRMSNorm):
    """Keep checkpoint deltas until FP32 (1+w), as official Clover does."""
    def __init__(self, size, eps):
        super().__init__(size, eps)
        from freetoken.kernel.triton.norm import gemma_rmsnorm, gemma_fused_add_rmsnorm
        self.rmsnorm = gemma_rmsnorm
        self.fused_add_rmsnorm = gemma_fused_add_rmsnorm


class CloverMoeModelLM(Qwen3_5ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model.norm = CloverPlusOneRMSNorm(config.hidden_size, config.rms_norm_eps)
        for layer in self.model.layers.op_list:
            layer.input_layernorm = CloverPlusOneRMSNorm(config.hidden_size, config.rms_norm_eps)
            layer.post_attention_layernorm = CloverPlusOneRMSNorm(config.hidden_size, config.rms_norm_eps)
            if not layer._is_linear:
                layer.self_attn.q_norm = CloverPlusOneRMSNorm(config.head_dim, config.rms_norm_eps)
                layer.self_attn.k_norm = CloverPlusOneRMSNorm(config.head_dim, config.rms_norm_eps)
                continue
            mixer = layer.linear_attn
            mixer.conv_right_weight = torch.empty(mixer.conv_dim, 1, mixer.conv_kernel_size - 1)
            mixer.bd_softmax_logit_scale = torch.empty(mixer.num_v_heads)
            mixer.bd_softmax_history_bias = torch.empty(mixer.num_v_heads)
            mixer.bd_softmax_history_c = torch.empty(mixer.num_v_heads)

    def forward(self):
        ctx = get_global_ctx()
        preview = getattr(ctx.batch, "clover_preview", False)
        if preview or getattr(ctx.batch, "clover_transactions", None) is not None:
            from .preview import block_forward
            return block_forward(self, ctx, preview=preview)
        return super().forward()
