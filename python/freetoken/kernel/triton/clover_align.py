"""Fixed-size expert grouping for short Clover blocks, without a slot-squared histogram."""

import torch
import triton
import triton.language as tl


@triton.jit
def _align(ids, sorted_ids, experts, total, N: tl.constexpr, TILE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    slot = tl.load(ids + index, index < N, other=2147483647).to(tl.int64)
    keys = (slot << 32) | index.to(tl.int64)
    keys = tl.sort(keys, descending=False)
    slot = keys >> 32
    previous = tl.gather(slot, tl.maximum(index - 1, 0), axis=0)
    following = tl.gather(slot, tl.minimum(index + 1, BLOCK - 1), axis=0)
    first = (index == 0) | (slot != previous)
    last = (index == N - 1) | (slot != following)
    start = tl.associative_scan(tl.where(first, index, 0), 0, _maximum)
    length = index - start + 1
    padding = tl.where(last & (index < N), tl.cdiv(length, TILE) * TILE - length, 0)
    position = index + tl.cumsum(padding) - padding
    tl.store(sorted_ids + position, keys.to(tl.int32), index < N)
    tl.store(experts + position // TILE, slot.to(tl.int32), (index < N) & (position % TILE == 0))
    tl.store(total, N + tl.sum(padding, 0))


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


def align_clover_routes(ids, tile, num_experts):
    n = ids.numel()
    if n > 1024:
        from freetoken.moe.fused import moe_align_block_size

        return moe_align_block_size(ids, tile, num_experts)
    sorted_ids = torch.full((n * tile,), n, dtype=torch.int32, device=ids.device)
    experts = torch.empty(n, dtype=torch.int32, device=ids.device)
    total = torch.empty(1, dtype=torch.int32, device=ids.device)
    _align[(1,)](ids, sorted_ids, experts, total, n, tile, triton.next_power_of_2(n))
    return sorted_ids, experts, total
