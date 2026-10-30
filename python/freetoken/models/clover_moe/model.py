"""Clover causal model with explicit storage for the read-only preview weights.

Architecture reference: DiffuSpace/Clover-35B-A3B-Preview,
4b0402b65e523d0183a14c431955438714cb16ff/modeling_clover_moe.py.
No HF implementation is copied here; causal operators reuse FreeToken's Qwen
path, leaving ordinary Qwen and Flash-Next arithmetic unchanged.
"""
import torch

from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM


class CloverMoeModelLM(Qwen3_5ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers.op_list:
            if not layer._is_linear:
                continue
            mixer = layer.linear_attn
            mixer.conv_right_weight = torch.empty(mixer.conv_dim, 1, mixer.conv_kernel_size - 1)
            mixer.bd_softmax_logit_scale = torch.empty(mixer.num_v_heads)
            mixer.bd_softmax_history_bias = torch.empty(mixer.num_v_heads)
            mixer.bd_softmax_history_c = torch.empty(mixer.num_v_heads)
