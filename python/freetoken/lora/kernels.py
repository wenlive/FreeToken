"""FP32 low-rank GEMV without per-route factor gathers or temporary casts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _shrink(X, A, IDS, LOW, K: tl.constexpr, R: tl.constexpr,
            SX0: tl.constexpr, SX1: tl.constexpr, SAE: tl.constexpr, SAR: tl.constexpr,
            SAK: tl.constexpr, TOP_K: tl.constexpr, EXPERT: tl.constexpr,
            ROUTED: tl.constexpr, BK: tl.constexpr):
    route, rank = tl.program_id(0), tl.program_id(1)
    row = route if ROUTED else route // TOP_K
    expert = tl.load(IDS + route).to(tl.int64) if EXPERT else 0
    k = tl.arange(0, BK)
    x = tl.load(X + row * SX0 + k * SX1, mask=k < K, other=0).to(tl.float32)
    a = tl.load(A + expert * SAE + rank * SAR + k * SAK, mask=k < K, other=0).to(tl.float32)
    tl.store(LOW + route * R + rank, tl.sum(x * a, axis=0))


@triton.jit
def _expand(LOW, B, IDS, BASE, OUT, N: tl.constexpr, R: tl.constexpr,
            SBE: tl.constexpr, SBN: tl.constexpr, SBR: tl.constexpr,
            SY0: tl.constexpr, SY1: tl.constexpr, SO0: tl.constexpr, SO1: tl.constexpr,
            EXPERT: tl.constexpr, HAS_BASE: tl.constexpr, SCALE: tl.constexpr,
            BN: tl.constexpr, BR: tl.constexpr):
    route = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    expert = tl.load(IDS + route).to(tl.int64) if EXPERT else 0
    low = tl.load(LOW + route * R + r, mask=r < R, other=0)
    b = tl.load(B + expert * SBE + n[:, None] * SBN + r[None, :] * SBR,
                mask=(n[:, None] < N) & (r[None, :] < R), other=0)
    value = tl.sum(b * low[None, :], axis=1) * SCALE
    if HAS_BASE:
        value += tl.load(BASE + route * SY0 + n * SY1, mask=n < N, other=0).to(tl.float32)
    tl.store(OUT + route * SO0 + n * SO1, value, mask=n < N)


def residual(x, a, b, *, ids=None, routed=True, base=None, scaling=1.0):
    """Use logical ids for expert factors and round only at the final output store."""
    expert = ids is not None
    routes, top_k = (ids.numel(), ids.shape[1]) if expert else (x.shape[0], 1)
    rank, width = a.shape[-2], b.shape[-2]
    low = torch.empty((routes, rank), device=x.device, dtype=torch.float32)
    output = torch.empty_like(base) if base is not None else torch.empty((routes, width), device=x.device, dtype=torch.float32)
    flat_ids = ids.reshape(-1) if expert else None
    _shrink[(routes, rank)](x, a, flat_ids, low, x.shape[-1], rank, x.stride(0), x.stride(1),
                           a.stride(0) if expert else 0, a.stride(-2), a.stride(-1), top_k,
                           expert, routed, triton.next_power_of_2(x.shape[-1]), num_warps=4)
    _expand[(routes, triton.cdiv(width, 64))](low, b, flat_ids, base, output, width, rank,
            b.stride(0) if expert else 0, b.stride(-2), b.stride(-1),
            base.stride(0) if base is not None else 0, base.stride(1) if base is not None else 0,
            output.stride(0), output.stride(1), expert, base is not None, scaling,
            64, triton.next_power_of_2(rank), num_warps=4)
    return output
