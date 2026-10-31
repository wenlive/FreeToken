"""Grouped MoE over native GGUF bytes; activation precision is GGML Q8_1.

Host orchestration follows FreeToken's existing fused_q4_0 path. Each projection
keeps its declared GGML type; weights are never requantized or expanded in HBM.
"""

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
        intermediate, down_q, topk_ids, 1, down_type, hidden, count * top_k
    )
    output = output.reshape(count, top_k, hidden)
    return (output * topk_weights[..., None].to(output.dtype)).sum(dim=1)
