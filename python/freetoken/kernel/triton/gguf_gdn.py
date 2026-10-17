"""GGUF GDN：循环 GQA 头布局、FP32 状态与按请求顺序更新。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent(Q, K, V, A, B, NEG_A, DT, STATE, CU, SLOTS, OUT,
               TRACK_BOUNDARY, TRACK_DST, HK: tl.constexpr, HV: tl.constexpr,
               D: tl.constexpr, QS: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
               KS: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
               VS: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
               AS: tl.constexpr, BS: tl.constexpr, TRACK_COUNT: tl.constexpr,
               DECODE: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    begin, end = tl.load(CU + req), tl.load(CU + req + 1)
    slot = tl.load(SLOTS + req).to(tl.int64)
    offset = tl.arange(0, D)
    state_offset = (slot * HV + head) * D * D + offset[:, None] * D + offset[None, :]
    state = tl.load(STATE + state_offset).to(tl.float32)
    negative_a = tl.load(NEG_A + head)
    dt = tl.load(DT + head)
    # GGUF value heads 为循环排列，Q/K 保持原顺序；不使用 HF 的 contiguous repeat。
    q_head = head % HK
    for row in range(begin, end):
        # prefill 卷积返回 channels-first 的转置视图，head/feature 不一定连续。
        q = tl.load(Q + row * QS + q_head * QH + offset * QD).to(tl.float32)
        k = tl.load(K + row * KS + q_head * KH + offset * KD).to(tl.float32)
        v = tl.load(V + row * VS + head * VH + offset * VD).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6)
        k *= tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
        a = tl.load(A + row * AS + head).to(tl.float32) + dt
        b = tl.load(B + row * BS + head).to(tl.float32)
        softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        decay = tl.exp(negative_a * softplus)
        beta = 1.0 / (1.0 + tl.exp(-b))
        kv = tl.sum(state * k[:, None], 0)
        delta = (v - decay * kv) * beta
        state = decay * state + k[:, None] * delta[None, :]
        output = tl.sum(state * q[:, None], 0) * (D ** -.5)
        tl.store(OUT + (row * HV + head) * D + offset, output)
        if not DECODE:
            for track in tl.static_range(TRACK_COUNT):
                if row + 1 == tl.load(TRACK_BOUNDARY + track):
                    dst = tl.load(TRACK_DST + track).to(tl.int64)
                    dst_offset = (dst * HV + head) * D * D + offset[:, None] * D + offset[None, :]
                    tl.store(STATE + dst_offset, state)
    tl.store(STATE + state_offset, state)


def gguf_gdn_forward(q, k, v, a, b, negative_a, dt_bias, state, fla, *, is_decode):
    """decode/prefill 共用 FP32 递推，直接写 live 与 track slot，保留池的 COW 生命周期。"""
    if any(t.dtype != torch.float32 for t in (q, k, v, a, b, negative_a, dt_bias, state)):
        raise ValueError("GGUF GDN requires FP32 activation, parameters and recurrent state")
    tensors = (q,k,v,a,b,negative_a,dt_bias,state)
    if any(not t.is_cuda or t.device != q.device for t in tensors):
        raise ValueError("GGUF GDN tensors must share one CUDA device")
    if q.ndim != 3 or v.ndim != 3 or state.ndim != 4:
        raise ValueError("incompatible GGUF GDN tensor ranks")
    total, hk, dim = q.shape
    hv = v.shape[1]
    if (hk <= 0 or hv <= 0 or k.shape != q.shape or v.shape != (total,hv,dim)
            or a.shape != (total,hv) or b.shape != (total,hv)
            or negative_a.shape != (hv,) or dt_bias.shape != (hv,) or hv % hk
            or state.shape[1:] != (hv, dim, dim) or dim <= 0 or dim & (dim - 1)
            or not state.is_contiguous() or not negative_a.is_contiguous()
            or not dt_bias.is_contiguous() or a.stride(-1) != 1 or b.stride(-1) != 1):
        raise ValueError("incompatible GGUF GDN state/head geometry")
    output = torch.empty((total, hv, dim), dtype=torch.float32, device=q.device)
    if getattr(fla, 'compact_mtp_pool', None) is not None:
        from .gguf_gdn_compact import verify
        return verify(q, k, v, a, b, negative_a, dt_bias, state, fla, output, is_decode=is_decode)
    track_count = 0 if fla.track_dst is None else fla.track_dst.numel()
    _recurrent[(fla.cache_indices.numel(), hv)](
        q, k, v, a, b, negative_a, dt_bias, state, fla.cu_seqlens, fla.cache_indices, output,
        fla.track_boundary_row, fla.track_dst, hk, hv, dim, *q.stride(), *k.stride(),
        *v.stride(), a.stride(0), b.stride(0), track_count, is_decode,
        num_warps=8, enable_fp_fusion=False)
    return output
