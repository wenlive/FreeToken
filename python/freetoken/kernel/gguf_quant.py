"""GGUF 压缩矩阵和稀疏行的 Triton 算子，输出/累积保持 FP32。

块布局依据冻结 Strata 的 artifact/dequant.hpp（ggml 3cf03257）；
解量化只发生在寄存器/调用方要求的稀疏行，不创建常驻 BF16 主干副本。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _u16(P, offset, mask):
    lo = tl.load(P + offset, mask, 0).to(tl.uint32)
    hi = tl.load(P + offset + 1, mask, 0).to(tl.uint32)
    return (lo | hi << 8).to(tl.uint16)


@triton.jit
def _half(P, offset, mask):
    return _u16(P, offset, mask).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _iq4(code):
    # 查表用 select，避免为每层创建重复 GPU 常量张量。
    value = tl.where(code == 0, -127, -104)
    value = tl.where(code == 2, -83, value)
    value = tl.where(code == 3, -65, value)
    value = tl.where(code == 4, -49, value)
    value = tl.where(code == 5, -35, value)
    value = tl.where(code == 6, -22, value)
    value = tl.where(code == 7, -10, value)
    value = tl.where(code == 8, 1, value)
    value = tl.where(code == 9, 13, value)
    value = tl.where(code == 10, 25, value)
    value = tl.where(code == 11, 38, value)
    value = tl.where(code == 12, 53, value)
    value = tl.where(code == 13, 69, value)
    value = tl.where(code == 14, 89, value)
    return tl.where(code == 15, 113, value).to(tl.float32)


@triton.jit
def _decode(P, row, col, mask, K: tl.constexpr, TYPE: tl.constexpr,
            BLOCK: tl.constexpr, BYTES: tl.constexpr):
    """以逻辑行列定位压缩块；所有 scale 先转 FP32，避免 BF16 二次舍入。"""
    base = row * (K // BLOCK * BYTES) + col // BLOCK * BYTES
    j = col % BLOCK
    if TYPE == 0:
        a = _u16(P, base, mask).to(tl.uint32)
        b = _u16(P, base + 2, mask).to(tl.uint32)
        value = (a | (b << 16)).to(tl.float32, bitcast=True)
    elif TYPE == 1:
        value = _half(P, base, mask)
    elif TYPE == 30:
        value = _u16(P, base, mask).to(tl.bfloat16, bitcast=True).to(tl.float32)
    elif TYPE == 42:
        q = tl.load(P + base + 2 + j // 4, mask, 0).to(tl.int32)
        value = (((q >> (2 * (j % 4))) & 3) - 1).to(tl.float32) * _half(P, base, mask)
    elif TYPE == 8:
        q = tl.load(P + base + 2 + j, mask, 0).to(tl.int8, bitcast=True).to(tl.float32)
        value = q * _half(P, base, mask)
    elif TYPE == 2 or TYPE == 6 or TYPE == 20:
        if TYPE == 6:
            q = tl.load(P + base + 6 + j % 16, mask, 0).to(tl.int32)
            high = tl.load(P + base + 2 + j // 8, mask, 0).to(tl.int32)
            code = ((q >> (4 * (j // 16))) & 15) | (((high >> (j % 8)) & 1) << 4)
            v = (code - 16).to(tl.float32)
        else:
            q = tl.load(P + base + 2 + j % 16, mask, 0).to(tl.int32)
            code = (q >> (4 * (j // 16))) & 15
            if TYPE == 20:
                v = _iq4(code)
            else:
                v = (code - 8).to(tl.float32)
        value = v * _half(P, base, mask)
    elif TYPE == 12 or TYPE == 13:
        group = j // 32
        lo = tl.load(P + base + 4 + tl.where(group < 4, group, group + 4), mask, 0).to(tl.int32)
        hi_d = tl.load(P + base + 4 + tl.where(group < 4, group, group - 4), mask, 0).to(tl.int32)
        hi_m = tl.load(P + base + 4 + tl.where(group < 4, group + 4, group), mask, 0).to(tl.int32)
        scale = tl.where(group < 4, lo & 63, (lo & 15) | ((hi_d >> 6) << 4))
        minimum = tl.where(group < 4, hi_m & 63, (lo >> 4) | ((hi_m >> 6) << 4))
        if TYPE == 13:
            q = tl.load(P + base + 48 + j // 64 * 32 + j % 32, mask, 0).to(tl.int32)
            high = tl.load(P + base + 16 + j % 32, mask, 0).to(tl.int32)
            code = ((q >> (4 * (group % 2))) & 15) | (((high >> group) & 1) << 4)
        else:
            q = tl.load(P + base + 16 + j // 64 * 32 + j % 32, mask, 0).to(tl.int32)
            code = (q >> (4 * (group % 2))) & 15
        value = _half(P, base, mask) * scale.to(tl.float32) * code.to(tl.float32)
        value -= _half(P, base + 2, mask) * minimum.to(tl.float32)
    elif TYPE == 11:
        group = j // 16
        low = tl.load(P + base + 96 + group % 8, mask, 0).to(tl.int32)
        high = tl.load(P + base + 104 + group % 4, mask, 0).to(tl.int32)
        scale = (((low >> (4 * (group // 8))) & 15) | (((high >> (2 * (group // 4))) & 3) << 4)) - 32
        q = tl.load(P + base + 32 + j // 128 * 32 + j % 32, mask, 0).to(tl.int32)
        hm = tl.load(P + base + j % 32, mask, 0).to(tl.int32)
        code = ((q >> (2 * ((j % 128) // 32))) & 3) - tl.where(((hm >> (j // 32)) & 1) != 0, 0, 4)
        value = _half(P, base + 108, mask) * scale.to(tl.float32) * code.to(tl.float32)
    elif TYPE == 14:
        low = tl.load(P + base + j // 128 * 64 + j % 32 + (j % 64 // 32) * 32, mask, 0).to(tl.int32)
        high = tl.load(P + base + 128 + j // 128 * 32 + j % 32, mask, 0).to(tl.int32)
        scale = tl.load(P + base + 192 + j // 16, mask, 0).to(tl.int8, bitcast=True).to(tl.float32)
        code = (((low >> (4 * ((j % 128) // 64))) & 15) | (((high >> (2 * ((j % 128) // 32))) & 3) << 4)) - 32
        value = _half(P, base + 208, mask) * scale * code.to(tl.float32)
    elif TYPE == 23:
        group = j // 32
        low = tl.load(P + base + 4 + group // 2, mask, 0).to(tl.int32)
        high = _u16(P, base + 2, mask).to(tl.int32)
        scale = (((low >> (4 * (group % 2))) & 15) | (((high >> (2 * group)) & 3) << 4)) - 32
        q = tl.load(P + base + 8 + group * 16 + j % 16, mask, 0).to(tl.int32)
        code = (q >> (4 * ((j % 32) // 16))) & 15
        value = _half(P, base, mask) * scale.to(tl.float32) * _iq4(code)
    return tl.where(mask, value, 0)


@triton.jit
def _rows(P, IDS, OUT, COUNT: tl.constexpr, K: tl.constexpr, TYPE: tl.constexpr,
          BLOCK: tl.constexpr, BYTES: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    index = tl.load(IDS + row)
    value = _decode(P, index, col, col < K, K, TYPE, BLOCK, BYTES)
    tl.store(OUT + row * K + col, value, col < K)


@triton.jit
def _q8_activation(X, OUT, T: tl.constexpr, K: tl.constexpr, XS0: tl.constexpr,
                   XS1: tl.constexpr, HALF_SCALE: tl.constexpr):
    token, block = tl.program_id(0), tl.program_id(1)
    cols = block * 32 + tl.arange(0, 32)
    x = tl.load(X + token * XS0 + cols * XS1).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    d = tl.minimum(amax / 127.0, 65504.0) if HALF_SCALE else amax / 127.0
    inv = tl.where(amax == 0, 0.0, 1.0 / d)
    z = x * inv
    q = tl.minimum(tl.maximum((z + tl.where(z >= 0, .5, -.5)).to(tl.int32), -127), 127)
    ds = d.to(tl.float16).to(tl.float32) if HALF_SCALE else d
    tl.store(OUT + token * K + cols, q.to(tl.float32) * ds)


@triton.jit
def _gemv(P, X, OUT, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          TYPE: tl.constexpr, BLOCK: tl.constexpr, BYTES: tl.constexpr,
          XS0: tl.constexpr, XS1: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr,
          BK: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(1)
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    col = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for begin in range(triton.cdiv(K, BK)):
        k = begin * BK + col
        w = _decode(P, row[:, None], k[None, :], (row[:, None] < N) & (k[None, :] < K), K, TYPE, BLOCK, BYTES)
        x = tl.load(X + token * XS0 + k * XS1, k < K, 0).to(tl.float32)
        acc += w * x[None, :]
    value = tl.sum(acc, 1)
    tl.store(OUT + token * OS0 + row * OS1, value, row < N)


@triton.jit
def _gemm(P, X, OUT, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          TYPE: tl.constexpr, BLOCK: tl.constexpr, BYTES: tl.constexpr,
          XS0: tl.constexpr, XS1: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """prefill 按 tile 解量化，用 TF32x3 tensor core 保持 FP32 乘加精度。"""
    tokens = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for begin in range(triton.cdiv(K, BK)):
        k = begin * BK + cols
        a = tl.load(X + tokens[:, None] * XS0 + k[None, :] * XS1,
                    (tokens[:, None] < T) & (k[None, :] < K), 0).to(tl.float32)
        w = _decode(P, rows[:, None], k[None, :], (rows[:, None] < N) & (k[None, :] < K),
                    K, TYPE, BLOCK, BYTES)
        acc += tl.dot(a, tl.trans(w), input_precision="tf32x3")
    tl.store(OUT + tokens[:, None] * OS0 + rows[None, :] * OS1, acc,
             (tokens[:, None] < T) & (rows[None, :] < N))


@triton.jit
def _affine_sum_correction(P, ORIGINAL, QUANTIZED, OUT, K: tl.constexpr, N: tl.constexpr,
                           BYTES: tl.constexpr, BIAS: tl.constexpr,
                           XS0: tl.constexpr, XS1: tl.constexpr,
                           OS0: tl.constexpr, OS1: tl.constexpr, BG: tl.constexpr):
    """Q4_0/Q5_0 的 offset 消费 Q8_1 原始输入 sum，不能用量化整数 sum 替代。"""
    token, row = tl.program_id(0), tl.program_id(1)
    group = tl.arange(0, BG)
    col = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    original = tl.load(ORIGINAL + token * XS0 + col * XS1, col < K, 0).to(tl.float32)
    quantized = tl.load(QUANTIZED + token * K + col, col < K, 0)
    total = tl.sum(original, 1)
    total = tl.minimum(tl.maximum(total, -65504.0), 65504.0).to(tl.float16).to(tl.float32)
    offset = row * (K // 32) * BYTES + group * BYTES
    d = _half(P, offset, group < K // 32)
    correction = tl.sum(d * BIAS * (tl.sum(quantized, 1) - total), 0)
    output = tl.load(OUT + token * OS0 + row * OS1)
    tl.store(OUT + token * OS0 + row * OS1, output + correction)


def _geometry(type_id):
    from freetoken.models.qwen4_exp.gguf_source import GGUF_TYPES
    _, block, size = GGUF_TYPES[type_id]
    return block, size


def dequantize_rows(packed, row_ids, width, type_id, out=None):
    """仅展开选中的行，供压缩 embedding 和 RowStore IQ4_NL 缓冲使用。"""
    block, size = _geometry(type_id)
    ids = row_ids.reshape(-1)
    if out is None:
        out = torch.empty((ids.numel(), width), dtype=torch.float32, device=packed.device)
    _rows[(ids.numel(), triton.cdiv(width, 256))](packed, ids, out, ids.numel(), width,
                                               type_id, block, size, 256, num_warps=4)
    return out


def quantize_activation(x, *, half_scale=True):
    """32 元素一组，round-away-from-zero；dense Q8_1 用 FP16 scale，专家用 FP32。"""
    if x.shape[-1] % 32:
        raise ValueError("Q8 activation width must be divisible by 32")
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _q8_activation[(x.shape[0], x.shape[1] // 32)](x, out, *x.shape, *x.stride(), half_scale,
                                                num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _gguf_plus_one_norm(X, W, OUT, H: tl.constexpr, EPS: tl.constexpr,
                        XS: tl.constexpr, OS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * XS + col, col < H, 0).to(tl.float32)
    gamma = 1.0 + tl.load(W + col, col < H, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    tl.store(OUT + row * OS + col, x * inv * gamma, col < H)


def gguf_gemma_rmsnorm(x, weight, eps=1e-6, out=None):
    """The GGUF Q/K norm has one fixed FP32 reduction for every batch size."""
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda or x.stride(-1) != 1:
        raise ValueError("GGUF norm requires a CUDA FP32 matrix with unit feature stride")
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype or out.stride(-1) != 1:
        raise ValueError("GGUF norm output geometry or precision differs from its input")
    _gguf_plus_one_norm[(x.shape[0],)](x,weight,out,x.shape[1],eps,x.stride(0),out.stride(0),
        triton.next_power_of_2(x.shape[1]),num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def _gguf_ple_gate(KEY, QUERY, OUT, H: tl.constexpr, HC: tl.constexpr,
                   KS: tl.constexpr, QS: tl.constexpr, BLOCK: tl.constexpr):
    token, group = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    key = tl.load(KEY + token * KS + group * H + col, col < H, 0).to(tl.float32)
    query = tl.load(QUERY + token * QS + group * H + col, col < H, 0).to(tl.float32)
    score = tl.sum(key * query, 0) / (H ** 0.5)
    sign = tl.where(score > 0, 1.0, tl.where(score < 0, -1.0, 0.0))
    magnitude = tl.sqrt(tl.maximum(tl.abs(score), 1.0e-6))
    gate = 1.0 / (1.0 + tl.exp(-sign * magnitude))
    tl.store(OUT + token * HC + group, gate)


def gguf_ple_gate(key, query, hc_count):
    """One fixed FP32 dot reduction per token/stream, independent of batch size."""
    if (key.ndim != 2 or query.shape != key.shape or hc_count <= 0
            or key.shape[1] % hc_count or key.dtype != torch.float32
            or query.dtype != torch.float32 or not key.is_cuda
            or key.device != query.device or key.stride(-1) != 1 or query.stride(-1) != 1):
        raise ValueError("GGUF PLE gate requires matching CUDA FP32 matrices and stream geometry")
    hidden = key.shape[1] // hc_count
    gate = torch.empty((key.shape[0], hc_count, 1), device=key.device, dtype=torch.float32)
    if key.shape[0]:
        _gguf_ple_gate[(key.shape[0], hc_count)](key,query,gate,hidden,hc_count,
            key.stride(0),query.stride(0),triton.next_power_of_2(hidden),
            num_warps=4,enable_fp_fusion=False)
    return gate


def compressed_linear(packed, x, rows, type_id, *, quantize_input=True, out=None, q8_source=None):
    """压缩权重直接参与 FP32 GEMV；BF16/F16/F32 权重保留 FP32 激活。"""
    block, size = _geometry(type_id)
    width = x.shape[-1]
    if quantize_input and type_id not in (0, 1, 30):
        q8_source = x
        x = quantize_activation(x)
    if out is None:
        out = torch.empty((x.shape[0], rows), dtype=torch.float32, device=x.device)
    if x.shape[0] == 0:
        return out
    # A fixed accumulation also covers single-token decode and cache-tail prefill.
    _gemm[(triton.cdiv(x.shape[0], 16), triton.cdiv(rows, 32))](packed, x, out, x.shape[0], rows, width,
        type_id, block, size, *x.stride(), *out.stride(), 16, 32, 64, num_warps=4,
        enable_fp_fusion=False)
    if type_id in (2, 6) and q8_source is not None:
        _affine_sum_correction[(x.shape[0], rows)](packed, q8_source, x, out, width, rows,
            size, 8 if type_id == 2 else 16, *q8_source.stride(), *out.stride(),
            triton.next_power_of_2(width // 32), num_warps=4, enable_fp_fusion=False)
    return out
