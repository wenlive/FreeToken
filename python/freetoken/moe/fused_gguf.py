"""Grouped MoE over native GGUF bytes; activation precision is GGML Q8_1.

Host orchestration follows FreeToken's existing fused_q4_0 path. Each projection
keeps its declared GGML type; weights are never requantized or expanded in HBM.
"""

import torch

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.gguf import ggml_moe_a8_vec, ggml_moe_a8, ggml_moe_get_block_size
from freetoken.layers.activation import silu_and_mul
from freetoken.kernel.triton.clover_align import align_clover_routes


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
    grouped = count > 1
    if grouped and gate_up_type in (2, 8, 12, 14) and down_type in (2, 8, 12, 14):
        block_size = ggml_moe_get_block_size(gate_up_type)
        sorted_ids, expert_ids, padded = align_clover_routes(topk_ids, block_size, gate_up_q.shape[0])
        gate_up = ggml_moe_a8(hidden_states, gate_up_q, sorted_ids, expert_ids, padded,
                              gate_up_type, gate_up_q.shape[1], top_k, count)
    else:
        grouped = False
        gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k,
                                 gate_up_type, gate_up_q.shape[1], count)
    intermediate = silu_and_mul(gate_up)
    if grouped:
        if ggml_moe_get_block_size(down_type) != block_size:
            sorted_ids, expert_ids, padded = align_clover_routes(
                topk_ids, ggml_moe_get_block_size(down_type), down_q.shape[0])
        output = ggml_moe_a8(intermediate.float(), down_q, sorted_ids, expert_ids, padded,
                             down_type, hidden, 1, count * top_k)
    else:
        output = ggml_moe_a8_vec(intermediate.float(), down_q, topk_ids, 1,
                                 down_type, hidden, count * top_k)
    # Clover rounds each FP32 down*router contribution once, then sums in FP32.
    # Casting down or router weights earlier loses information before routing.
    routed = (
        output.reshape(count, top_k, hidden) * topk_weights[..., None].float()
    ).to(hidden_states.dtype)
    result = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(routed, result)
    return result
