"""Q2_0 weight-only MoE: native blocks, BF16 activations, FP32 accumulation."""
from __future__ import annotations

import torch
import os
import triton
import triton.language as tl


@triton.jit
def _q2_values(W, base, row, k, N: tl.constexpr, K: tl.constexpr):
    # 每个 64 元素块保存 FP16 scale + 16 字节相邻 2-bit 编码，区别于 Q2_K/TQ2_0。
    block = base + row[:, None] * (K // 64 * 18) + (k[None, :] // 64) * 18
    mask = (row[:, None] < N) & (k[None, :] < K)
    low = tl.load(W + block, mask, 0).to(tl.uint16)
    high = tl.load(W + block + 1, mask, 0).to(tl.uint16)
    scale = (low | (high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    packed = tl.load(W + block + 2 + k[None, :] % 64 // 4, mask, 0).to(tl.int32)
    code = (packed >> ((k[None, :] % 4) * 2)) & 3
    # 权重先舍入到 BF16，主干激活不再额外量化；乘加使用 FP32。
    return ((code - 1).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _gate_up(X, W, IDS, A, H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
             BK: tl.constexpr, BN: tl.constexpr):
    route = tl.program_id(0)
    token = route // TOPK
    expert = tl.load(IDS + route).to(tl.int64)
    row = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    base = expert * (2 * I * (H // 64 * 18))
    xv = tl.load(X + token * H + k, k < H, 0).to(tl.float32)
    gate = tl.sum(_q2_values(W, base, row, k, 2 * I, H) * xv[None, :], 1).to(tl.bfloat16).to(tl.float32)
    up = tl.sum(_q2_values(W, base, row + I, k, 2 * I, H) * xv[None, :], 1).to(tl.bfloat16).to(tl.float32)
    value = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(A + route * I + row, value, row < I)


@triton.jit
def _down(A, W, IDS, R, H: tl.constexpr, I: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(IDS + route).to(tl.int64)
    row = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    base = expert * (H * (I // 64 * 18))
    av = tl.load(A + route * I + k, k < I, 0).to(tl.float32)
    value = tl.sum(_q2_values(W, base, row, k, H, I) * av[None, :], 1)
    tl.store(R + route * H + row, value, row < H)


@triton.jit
def _reduce(R, WEIGHTS, Y, H: tl.constexpr, TOPK: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr):
    token = tl.program_id(0)
    row = tl.program_id(1) * BH + tl.arange(0, BH)
    route = tl.arange(0, BT)
    w = tl.load(WEIGHTS + token * TOPK + route, route < TOPK, 0).to(tl.float32)
    values = tl.load(R + (token * TOPK + route[:, None]) * H + row[None, :],
                     (route[:, None] < TOPK) & (row[None, :] < H), 0).to(tl.float32)
    tl.store(Y + token * H + row, tl.sum(values * w[:, None], 0), row < H)


def fused_experts_q2_0(x, gate_up, down, topk_weights, topk_ids, *, is_prefill=False):
    """根据实际输入规模选择向量计算或按专家分组的矩阵计算。"""
    if x.dtype != torch.bfloat16 or gate_up.dtype != torch.uint8 or down.dtype != torch.uint8:
        raise ValueError("Q2_0 requires BF16 activations and native uint8 blocks")
    if not all(t.is_contiguous() and t.is_cuda for t in (x, gate_up, down, topk_weights, topk_ids)):
        raise ValueError("Q2_0 tensors must be contiguous CUDA tensors")
    tokens, hidden = x.shape
    intermediate = gate_up.shape[1] // 2
    topk = topk_ids.shape[1]
    if (hidden % 64 or intermediate % 64 or gate_up.shape[2] != hidden // 64 * 18
            or down.shape != (gate_up.shape[0], hidden, intermediate // 64 * 18)
            or topk_ids.shape[0] != tokens or topk_weights.shape != topk_ids.shape):
        raise ValueError("incompatible Q2_0 expert geometry or routing tensors")
    if is_prefill and tokens >= 128 and os.environ.get("FREETOKEN_Q2_GROUPED_PREFILL", "1") != "0":
        return grouped_experts_q2_0(x, gate_up, down, topk_weights, topk_ids)
    active = torch.empty((tokens * topk, intermediate), dtype=x.dtype, device=x.device)
    routed = torch.empty((tokens * topk, hidden), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    _gate_up[(tokens * topk, triton.cdiv(intermediate, 4))](
        x, gate_up, topk_ids, active, hidden, intermediate, topk,
        triton.next_power_of_2(hidden), 4, num_warps=8)
    _down[(tokens * topk, triton.cdiv(hidden, 4))](
        active, down, topk_ids, routed, hidden, intermediate,
        triton.next_power_of_2(intermediate), 4, num_warps=4)
    _reduce[(tokens, triton.cdiv(hidden, 128))](
        routed, topk_weights, out, hidden, topk, triton.next_power_of_2(topk), 128, num_warps=4)
    return out


@triton.jit
def _grouped_q2_gemm(X, W, SORTED, EXPERTS, PADDED, Y,
                     ROUTES: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     INPUT_TOPK: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr):
    # 同一专家的多个 token 共享一份权重 tile；仅在寄存器中解量化，不展开整层 BF16。
    block = tl.program_id(0)
    if block * BM >= tl.load(PADDED):
        return
    route = tl.load(SORTED + block * BM + tl.arange(0, BM))
    expert = tl.load(EXPERTS + block).to(tl.int64)
    row = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(triton.cdiv(K, BK)):
        cols = start * BK + k
        a = tl.load(X + (route[:, None] // INPUT_TOPK) * K + cols[None, :],
                    (route[:, None] < ROUTES) & (cols[None, :] < K), 0)
        b = _q2_values(W, expert * N * (K // 64 * 18), row, cols, N, K).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(Y + route[:, None] * N + row[None, :], acc,
             (route[:, None] < ROUTES) & (row[None, :] < N))


@triton.jit
def _grouped_gate_act(GU, A, I: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    route = index // I
    feature = index % I
    gate = tl.load(GU + route * (2 * I) + feature, index < TOTAL, 0).to(tl.float32)
    up = tl.load(GU + route * (2 * I) + I + feature, index < TOTAL, 0).to(tl.float32)
    tl.store(A + index, gate / (1.0 + tl.exp(-gate)) * up, index < TOTAL)


def grouped_experts_q2_0(x, gate_up, down, topk_weights, topk_ids):
    """复用 FreeToken 的 GPU 路由分组，以 BF16 tensor-core GEMM 执行大批输入。"""
    from freetoken.moe.fused import moe_align_block_size

    tokens, hidden = x.shape
    intermediate = gate_up.shape[1] // 2
    topk = topk_ids.shape[1]
    routes = tokens * topk
    bm, bn, bk = 16, 32, 64
    sorted_ids, experts, padded = moe_align_block_size(topk_ids, bm, gate_up.shape[0])
    gate_values = torch.empty((routes, 2 * intermediate), dtype=x.dtype, device=x.device)
    active = torch.empty((routes, intermediate), dtype=x.dtype, device=x.device)
    routed = torch.empty((routes, hidden), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    _grouped_q2_gemm[(triton.cdiv(sorted_ids.numel(), bm), triton.cdiv(2 * intermediate, bn))](
        x, gate_up, sorted_ids, experts, padded, gate_values,
        routes, 2 * intermediate, hidden, topk, bm, bn, bk, num_warps=4)
    _grouped_gate_act[(triton.cdiv(routes * intermediate, 256),)](
        gate_values, active, intermediate, routes * intermediate, 256)
    _grouped_q2_gemm[(triton.cdiv(sorted_ids.numel(), bm), triton.cdiv(hidden, bn))](
        active, down, sorted_ids, experts, padded, routed,
        routes, hidden, intermediate, 1, bm, bn, bk, num_warps=4)
    _reduce[(tokens, triton.cdiv(hidden, 128))](
        routed, topk_weights, out, hidden, topk, triton.next_power_of_2(topk), 128, num_warps=4)
    return out
