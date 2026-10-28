"""GGUF MTP 验证卷积及全部前缀快照；保留逐行 decode 的 FP32 算术。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _verify_conv(X, W, STATE, SLOTS, DST, BOUNDARY, OUT,
                 D: tl.constexpr, T: tl.constexpr,
                 XS: tl.constexpr, XD: tl.constexpr, WS: tl.constexpr, WD: tl.constexpr,
                 SS: tl.constexpr, SD: tl.constexpr, SK: tl.constexpr,
                 TRACKS: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = col < D
    slot = tl.load(SLOTS).to(tl.int64)
    base = slot * SS + col * SD
    h0 = tl.load(STATE + base, mask, 0)
    h1 = tl.load(STATE + base + SK, mask, 0)
    h2 = tl.load(STATE + base + 2 * SK, mask, 0)
    w0 = tl.load(W + col * WS, mask, 0)
    w1 = tl.load(W + col * WS + WD, mask, 0)
    w2 = tl.load(W + col * WS + 2 * WD, mask, 0)
    w3 = tl.load(W + col * WS + 3 * WD, mask, 0)
    for row in tl.static_range(T):
        x = tl.load(X + row * XS + col * XD, mask, 0)
        acc = tl.full((BLOCK,), 0, tl.float32)
        acc += h0 * w0
        acc += h1 * w1
        acc += h2 * w2
        acc += x * w3
        value = acc / (1 + tl.exp(-acc))
        tl.store(OUT + row * D + col, value, mask)
        h0 = h1
        h1 = h2
        h2 = x
        for track in tl.static_range(TRACKS):
            if tl.load(BOUNDARY + track) == row + 1:
                dst = tl.load(DST + track).to(tl.int64)
                target = dst * SS + col * SD
                tl.store(STATE + target, h0, mask)
                tl.store(STATE + target + SK, h1, mask)
                tl.store(STATE + target + 2 * SK, h2, mask)
    tl.store(STATE + base, h0, mask)
    tl.store(STATE + base + SK, h1, mask)
    tl.store(STATE + base + 2 * SK, h2, mask)


def verify_conv(x, weight, state, slots, dst, boundary):
    """每个 program 独占通道，在寄存器内滚动历史并保存所有合法前缀。

    调用者保证单请求 live slot 与事务快照 slot 独立；递推和已接受前缀提交
    仍由原状态池处理。输入和卷积权重可以是模型投影产生的非连续视图。
    """
    if weight.shape[1] != 4 or state.shape[-1] != 3:
        raise ValueError('GGUF verify convolution requires width4/state3')
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    tracks = 0 if dst is None else dst.numel()
    _verify_conv[(triton.cdiv(x.shape[1], 128),)](x, weight, state, slots, dst, boundary, out,
        x.shape[1], len(x), *x.stride(), *weight.stride(), *state.stride(), tracks, 128,
        num_warps=4, num_stages=2)
    return out
