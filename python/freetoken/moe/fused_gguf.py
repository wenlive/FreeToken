"""Grouped MoE over native GGUF bytes; activation precision is GGML Q8_1.

Host orchestration follows FreeToken's existing fused_q4_0 path. Each projection
keeps its declared GGML type; weights are never requantized or expanded in HBM.
"""

import torch

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.gguf import ggml_moe_a8_vec
from freetoken.layers.activation import silu_and_mul


def fused_experts_gguf(
    hidden_states,
    gate_up_q,
    down_q,
    topk_weights,
    topk_ids,
    activation,
    *,
    gate_up_type,
    down_type,
):
    if activation != "silu":
        raise NotImplementedError("Mixed Clover GGUF currently requires SwiGLU")
    count, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    gate_up = ggml_moe_a8_vec(
        hidden_states,
        gate_up_q,
        topk_ids,
        top_k,
        gate_up_type,
        gate_up_q.shape[1],
        count,
    )
    intermediate = silu_and_mul(gate_up)
    output = ggml_moe_a8_vec(
        intermediate.float(), down_q, topk_ids, 1, down_type, hidden, count * top_k
    )
    # Clover rounds each FP32 down*router contribution once, then sums in FP32.
    # Casting down or router weights earlier loses information before routing.
    routed = (
        output.reshape(count, top_k, hidden) * topk_weights[..., None].float()
    ).to(hidden_states.dtype)
    result = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(routed, result)
    return result
