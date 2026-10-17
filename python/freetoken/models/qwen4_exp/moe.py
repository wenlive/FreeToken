from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config, layer_id=None, *, prefix=""):
        """GGUF router 使用压缩线性接口，保持 BF16 权重配 FP32 激活的直接计算。"""
        super().__init__(config, layer_id, prefix=prefix)
        self._gguf = getattr(config.quant, "dialect", None) == "gguf_strata"
        if self._gguf:
            from freetoken.layers import LinearReplicated
            self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False,
                                          quant_config=config.quant, prefix=f"{prefix}.gate")
            self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False,
                                                       quant_config=config.quant, prefix=f"{prefix}.shared_expert_gate")

    def _shared_gguf(self, hidden_states):
        """共享专家和门控在routed分支可能改写输入之前完成。"""
        shared = self.shared_expert.forward(hidden_states)
        gate = self.shared_expert_gate.forward(hidden_states).sigmoid().view(-1)
        return shared, gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """永久Q2的单token decode重叠搬运；prefill、验证窗口和其他模型沿用既有顺序。"""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        cache = getattr(self.experts, 'offload_cache', None)
        if (self._gguf and get_global_ctx().batch.is_decode
                and getattr(cache, 'decode_copy_stream', None) is not None):
            routed, (shared, gate) = self.experts.decode_with_shared(
                hidden_states, router_logits, self._shared_gguf)
            return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)
        shared = self.shared_expert.forward(hidden_states)
        if self._gguf:
            gate = self.shared_expert_gate.forward(hidden_states).sigmoid().view(-1)
        else:
            gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
