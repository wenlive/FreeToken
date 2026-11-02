"""Split the prefix KV scan across CTAs, then merge FP32 online-softmax states."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(Q, KC, VC, K, V, INDICES, PREFIX, PARTS, LSE,
              QS: tl.constexpr, QH: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
              VS: tl.constexpr, VH: tl.constexpr, HEADS: tl.constexpr,
              KV_HEADS: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
              SPLITS: tl.constexpr, PREVIEW: tl.constexpr, SCALE: tl.constexpr,
              M: tl.constexpr, N: tl.constexpr):
    head, split = tl.program_id(0), tl.program_id(1)
    kv_head = head // (HEADS // KV_HEADS)
    row, dim = tl.arange(0, M), tl.arange(0, D)
    prefix = tl.load(PREFIX)
    total = prefix + W
    span = tl.cdiv(tl.cdiv(total, SPLITS), N) * N
    first, end = split * span, tl.minimum((split + 1) * span, total)
    q = tl.load(Q + row[:, None] * QS + head * QH + dim[None, :], row[:, None] < W, other=0)
    maximum = tl.full((M,), -float('inf'), tl.float32)
    normalizer = tl.zeros((M,), tl.float32)
    accumulator = tl.zeros((M, D), tl.float32)
    for start in tl.range(first, end, N):
        col = start + tl.arange(0, N)
        past = (col < prefix) & (col < end)
        current = (col >= prefix) & (col < end)
        slot = tl.load(INDICES + col, past, other=0).to(tl.int64)
        cached_k = tl.load(KC + slot[None, :] * (KV_HEADS * D) + kv_head * D + dim[:, None], past[None, :], other=0)
        live_k = tl.load(K + (col[None, :] - prefix) * KS + kv_head * KH + dim[:, None], current[None, :], other=0)
        key = tl.where(past[None, :], cached_k, live_k)
        scores = tl.dot(q, key)
        allowed = (col[None, :] < end) & (row[:, None] < W)
        if not PREVIEW:
            allowed &= col[None, :] <= prefix + row[:, None]
        scores = tl.where(allowed, scores, -float('inf'))
        scale_log2 = SCALE * 1.4426950408889634
        new_max = tl.maximum(maximum, tl.max(scores, 1) * scale_log2)
        alpha = tl.where(new_max == -float('inf'), 0., tl.exp2(maximum - new_max))
        probabilities = tl.where(scores == -float('inf'), 0., tl.exp2(scores * scale_log2 - new_max[:, None]))
        cached_v = tl.load(VC + slot[:, None] * (KV_HEADS * D) + kv_head * D + dim[None, :], past[:, None], other=0)
        live_v = tl.load(V + (col[:, None] - prefix) * VS + kv_head * VH + dim[None, :], current[:, None], other=0)
        value = tl.where(past[:, None], cached_v, live_v)
        accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(value.dtype), value)
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        maximum = new_max
    output = accumulator / tl.where(normalizer > 0, normalizer, 1.)[:, None]
    lse = tl.where(normalizer > 0, maximum + tl.log2(normalizer), -float('inf'))
    tl.store(PARTS + ((head * SPLITS + split) * W + row[:, None]) * D + dim[None, :], output, row[:, None] < W)
    tl.store(LSE + (head * SPLITS + split) * W + row, lse, row < W)


@triton.jit
def _merge(PARTS, LSE, OUT, HEADS: tl.constexpr, D: tl.constexpr,
           W: tl.constexpr, SPLITS: tl.constexpr):
    head, row = tl.program_id(0), tl.program_id(1)
    split, dim = tl.arange(0, SPLITS), tl.arange(0, D)
    lse = tl.load(LSE + (head * SPLITS + split) * W + row)
    maximum = tl.max(lse, 0)
    weights = tl.exp2(lse - maximum)
    values = tl.load(PARTS + ((head * SPLITS + split[:, None]) * W + row) * D + dim[None, :])
    output = tl.sum(values * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(OUT + (row * HEADS + head) * D + dim, output)


def split_kv_attention(q, keys, values, indices, prefix_lens, k, v, *, preview, splits=8):
    width, heads, dim = q.shape
    assert width in (8, 16, 32) and dim in (64, 128, 256)
    assert splits in (2, 4, 8, 16)
    assert q.stride(2) == k.stride(2) == v.stride(2) == 1
    assert keys.is_contiguous() and values.is_contiguous() and indices.is_contiguous()
    assert prefix_lens.numel() == 1 and heads % keys.shape[1] == 0
    assert q.dtype == keys.dtype == values.dtype == k.dtype == v.dtype
    assert q.dtype in (torch.bfloat16, torch.float16)
    output = torch.empty((width, heads, dim), dtype=q.dtype, device=q.device)
    parts = torch.empty((heads, splits, width, dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((heads, splits, width), dtype=torch.float32, device=q.device)
    _partials[(heads, splits)](q, keys, values, k, v, indices, prefix_lens, parts, lse,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),
        heads,keys.shape[1],dim,width,splits,preview,dim**-.5,max(16,width),32,
        num_warps=4,num_stages=1)
    _merge[(heads,width)](parts,lse,output,heads,dim,width,splits,num_warps=4)
    return output
