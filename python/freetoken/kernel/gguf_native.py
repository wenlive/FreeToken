"""GGUF GPU vector kernels through FreeToken's versioned AOT/JIT module loader."""
from functools import lru_cache

import torch
import triton
import triton.language as tl

from .utils import KERNEL_PATH, load_jit

NATIVE_TYPES = frozenset((2, 6, 8, 11, 12, 13, 14, 20, 23, 42))


@lru_cache(maxsize=1)
def _module(*, build_directory=None):
    return load_jit(
        "gguf_mmvq_r1", cuda_files=["gguf_mmvq.cuh"],
        cuda_wrappers=[("quantize", "GGUFNativeMMVQ::quantize"), ("project", "GGUFNativeMMVQ::project")],
        extra_cuda_cflags=["--use_fast_math"],
        extra_include_paths=[str(KERNEL_PATH / "third_party/strata_mmvq/include")],
        build_directory=build_directory,
    )


def quantize_q8_1(x):
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda or x.shape[1] % 32:
        raise ValueError("Native Q8_1 requires a CUDA FP32 matrix of width divisible by 32")
    q = torch.empty((len(x), x.shape[1] // 32, 36), device=x.device, dtype=torch.uint8)
    if len(x):
        with torch.cuda.device(x.device):
            _module().quantize(x.contiguous(), q, torch.cuda.current_stream(x.device).cuda_stream)
    return q


def mmvq(packed, q8, rows, type_id, *, out=None):
    if type_id not in NATIVE_TYPES:
        raise ValueError("Native MMVQ does not implement this GGUF format")
    if q8.ndim != 3 or q8.shape[2] != 36 or q8.dtype != torch.uint8 or not q8.is_cuda:
        raise ValueError("Native MMVQ requires a packed CUDA Q8_1 matrix")
    if rows <= 0:
        raise ValueError("Native MMVQ output width must be positive")
    if out is None:
        out = torch.empty((len(q8), rows), device=q8.device, dtype=torch.float32)
    if out.shape != (len(q8), rows) or out.dtype != torch.float32 or out.device != q8.device:
        raise ValueError("Native MMVQ output shape, device or precision differs")
    if not len(q8):
        return out
    target = out if out.is_contiguous() else torch.empty((len(q8), rows), device=q8.device, dtype=torch.float32)
    with torch.cuda.device(q8.device):
        _module().project(packed, q8.contiguous(), target, type_id, q8.shape[1] * 32,
                          torch.cuda.current_stream(q8.device).cuda_stream)
        if target is not out:
            out.copy_(target)
    return out


@triton.jit
def _bf16_rows(W0, W1, X, Y, N0: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               XS0: tl.constexpr, XS1: tl.constexpr, YS0: tl.constexpr, YS1: tl.constexpr,
               BK: tl.constexpr):
    row, token = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BK)
    w = tl.where(row < N0, W0 + row * K, W1 + (row - N0) * K)
    acc = tl.full((BK,), 0., tl.float32)
    for first in range(triton.cdiv(K, BK)):
        ks = first * BK + cols
        weight = tl.load(w + ks, ks < K, 0).to(tl.float32)
        x = tl.load(X + token * XS0 + ks * XS1, ks < K, 0).to(tl.float32)
        acc += weight * x
    tl.store(Y + token * YS0 + row * YS1, tl.sum(acc, 0))


@triton.jit
def _bf16_row_tiles(W0, W1, X, Y, N0: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    XS0: tl.constexpr, XS1: tl.constexpr, YS0: tl.constexpr, YS1: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    token = tl.program_id(1)
    cols = tl.arange(0, BK)
    base = tl.where(row < N0, W0 + row * K, W1 + (row - N0) * K)
    acc = tl.full((BN, BK), 0., tl.float32)
    for first in range(triton.cdiv(K, BK)):
        ks = first * BK + cols
        weight = tl.load(base[:, None] + ks[None, :],
                         (row[:, None] < N) & (ks[None, :] < K), 0).to(tl.float32)
        value = tl.load(X + token * XS0 + ks * XS1, ks < K, 0)
        acc += weight * value[None, :]
    tl.store(Y + token * YS0 + row * YS1, tl.sum(acc, 1), row < N)


@triton.jit
def _bf16_batch(W0, W1, X, Y, N0: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                XS0: tl.constexpr, XS1: tl.constexpr, YS0: tl.constexpr, YS1: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    base = tl.where(col < N0, W0 + col * K, W1 + (col - N0) * K)
    acc = tl.zeros((BM, BN), tl.float32)
    for first in range(triton.cdiv(K, BK)):
        ks = first * BK + inner
        value = tl.load(X + row[:, None] * XS0 + ks[None, :] * XS1,
                        (row[:, None] < M) & (ks[None, :] < K), 0)
        weight = tl.load(base[None, :] + ks[:, None],
                         (col[None, :] < N) & (ks[:, None] < K), 0).to(tl.float32)
        # FP32输入不转BF16；tf32x3使用三次乘积保留高精度，累加及输出仍为FP32。
        acc += tl.dot(value, weight, input_precision='tf32x3')
    tl.store(Y + row[:, None] * YS0 + col[None, :] * YS1, acc,
             (row[:, None] < M) & (col[None, :] < N))


def bf16_gemv(packed, x, rows, *, out=None, packed1=None, first_rows=None):
    """按真实几何选择GEMV或批量权重复用，保留两段压缩源、FP32激活和任意有效stride。"""
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda or rows <= 0:
        raise ValueError("BF16 GEMV requires a CUDA FP32 matrix and positive output width")
    if out is None:
        out = torch.empty((len(x), rows), device=x.device, dtype=torch.float32)
    if out.shape != (len(x), rows) or out.dtype != torch.float32 or out.device != x.device:
        raise ValueError("BF16 GEMV output geometry differs")
    if not len(x):
        return out
    n0 = rows if first_rows is None else first_rows
    if not 0 < n0 <= rows:
        raise ValueError("BF16 GEMV segment split is invalid")
    p1 = packed if packed1 is None else packed1
    if (packed.dtype != torch.uint8 or packed.device != x.device or not packed.is_contiguous()
            or packed.numel() != n0 * x.shape[1] * 2
            or p1.dtype != torch.uint8 or p1.device != x.device or not p1.is_contiguous()
            or (n0 < rows and (packed1 is None or p1.numel() != (rows - n0) * x.shape[1] * 2))):
        raise ValueError("BF16 GEMV packed segment geometry differs")
    width = x.shape[1]
    w0, w1 = packed.view(torch.bfloat16), p1.view(torch.bfloat16)
    if len(x) >= 32 and rows >= 256:
        bm, bn = (32, 64) if width <= 512 else (16, 32)
        _bf16_batch[(triton.cdiv(len(x), bm), triton.cdiv(rows, bn))](
            w0, w1, x, out, n0, len(x), rows, width, *x.stride(), *out.stride(),
            bm, bn, 32, num_warps=4)
    elif width <= 512 and rows >= 1024:
        _bf16_row_tiles[(triton.cdiv(rows, 8), len(x))](
            w0, w1, x, out, n0, rows, width, *x.stride(), *out.stride(),
            8, 256, num_warps=4, enable_fp_fusion=False)
    else:
        _bf16_rows[(rows, len(x))](w0, w1, x, out,
            n0, rows, width, *x.stride(), *out.stride(), 256, num_warps=4, enable_fp_fusion=False)
    return out
