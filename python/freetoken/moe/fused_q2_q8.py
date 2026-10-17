"""GGUF 专家 Q2×Q8 路径；bank/cache 字节不变，激活及状态保留 FP32。"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Q, S, K: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr):
    token, block = tl.program_id(0), tl.program_id(1)
    col = block * 32 + tl.arange(0, 32)
    x = tl.load(X + token * XS0 + col * XS1).to(tl.float32)
    d = tl.max(tl.abs(x), 0) / 127.0
    inv = tl.where(d > 0, 1.0 / d, 0.0)
    z = x * inv
    q = (z + tl.where(z >= 0, .5, -.5)).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(Q + token * K + col, q)
    tl.store(S + token * (K // 32) + block, d)


@triton.jit
def _codes(W, expert, rows, cols, N: tl.constexpr, K: tl.constexpr):
    offset = expert * N * (K // 64 * 18) + rows * (K // 64 * 18) + (cols // 64) * 18
    valid = (rows < N) & (cols < K)
    lo = tl.load(W + offset, valid, 0).to(tl.uint16)
    hi = tl.load(W + offset + 1, valid, 0).to(tl.uint16)
    d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    byte = tl.load(W + offset + 2 + cols % 64 // 4, valid, 0).to(tl.int32)
    q = (((byte >> ((cols % 4) * 2)) & 3) - 1).to(tl.int8)
    return q, d


@triton.jit
def _routed_vector(Q32,S,W,IDS,Y,K:tl.constexpr,N:tl.constexpr,
                 INPUT_TOPK:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
    route=tl.program_id(0)
    token=route//INPUT_TOPK
    expert=tl.load(IDS+route).to(tl.int64)
    rows=tl.program_id(1)*BN+tl.arange(0,BN)
    groups=tl.arange(0,BG)
    words=tl.arange(0,8)
    base=expert*N*(K//64*18)+rows[:,None]*(K//64*18)+(groups[None,:]//2)*18
    valid=(rows[:,None]<N)&(groups[None,:]<K//32)
    # Q2 blocks are 18 bytes: load headers as bytes, never assume int32 alignment.
    low=tl.load(W+base,valid,0).to(tl.uint16)
    high=tl.load(W+base+1,valid,0).to(tl.uint16)
    dw=(low|(high<<8)).to(tl.float16,bitcast=True).to(tl.float32)
    codes=tl.load(W+base[:,:,None]+2+(groups[None,:,None]%2)*8+words[None,None,:],valid[:,:,None],0).to(tl.uint32)
    packed=((codes&3)|(((codes>>2)&3)<<8)|(((codes>>4)&3)<<16)|(((codes>>6)&3)<<24)).to(tl.int32)
    x=tl.load(Q32+token*(K//4)+groups[:,None]*8+words[None,:],groups[:,None]<K//32,0).to(tl.int32)
    # The unsigned two-bit codes need the exact integer sum(x) correction.
    products=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',constraints='=r,r,r',
        args=[packed,x[None,:,:]],dtype=tl.int32,is_pure=True,pack=1)
    correction=tl.inline_asm_elementwise('dp4a.s32.s32 $0, 0x01010101, $1, 0;',constraints='=r,r',
        args=[x],dtype=tl.int32,is_pure=True,pack=1)
    dot=tl.sum(products-correction[None,:,:],2).to(tl.float32)
    dx=tl.load(S+token*(K//32)+groups,groups<K//32,0)
    value=tl.sum(dot*dw*dx[None,:],1)
    tl.store(Y+route*N+rows,value,rows<N)

def _routed_project(value,bank,result,ids,input_topk):
    q,scales=quantize_expert_activation(value)
    k,n=value.shape[1],result.shape[1]
    _routed_vector[(ids.numel(),triton.cdiv(n,4))](q.view(torch.int32),scales,bank,ids,result,
        k,n,input_topk,4,triton.next_power_of_2(k//32),num_warps=4,enable_fp_fusion=False)


@triton.jit
def _grouped(Q, S, W, SORTED, EXPERTS, PADDED, Y,
             ROUTES: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             INPUT_TOPK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    block = tl.program_id(0)
    if block * BM >= tl.load(PADDED):
        return
    routes = tl.load(SORTED + block * BM + tl.arange(0, BM))
    tokens = routes // INPUT_TOPK
    expert = tl.load(EXPERTS + block).to(tl.int64)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 32):
        k = group * 32 + cols
        a = tl.load(Q + tokens[:, None] * K + k[None, :], routes[:, None] < ROUTES, 0)
        b, unused = _codes(W, expert, rows[:, None], k[None, :], N, K)
        # int8 tensor core 的块内点积为整数；块外使用原生 FP16 W scale 和 FP32 X scale。
        dot = tl.dot(a, tl.trans(b)).to(tl.float32)
        dx = tl.load(S + tokens * (K // 32) + group, routes < ROUTES, 0)
        _, dw = _codes(W, expert, rows, tl.full((BN,), group * 32, tl.int32), N, K)
        acc += dot * dw[None, :] * dx[:, None]
    tl.store(Y + routes[:, None] * N + rows[None, :], acc,
             (routes[:, None] < ROUTES) & (rows[None, :] < N))


def quantize_expert_activation(x):
    """专家 scale 为 FP32，舍入遵循冻结 Strata 的 GPU Q8 scaled 路径。"""
    if x.ndim != 2 or x.shape[1] % 32 or not x.is_cuda or x.dtype != torch.float32:
        raise ValueError("GGUF expert Q8 expects a CUDA FP32 matrix with width divisible by 32")
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scales = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.float32)
    _quantize[(x.shape[0], x.shape[1] // 32)](x, q, scales, x.shape[1], *x.stride(),
                                           num_warps=4, enable_fp_fusion=False)
    return q, scales


def fused_experts_q2_q8(x, gate_up, down, topk_weights, topk_ids, *, is_prefill=False):
    """复用完整真实专家及 GPU 路由，只改变计算契约，不创建 BF16 专家副本。"""
    from freetoken.moe.fused_q2_0 import _grouped_gate_act, _reduce
    if x.ndim != 2 or gate_up.ndim != 3 or down.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("GGUF Q2xQ8 expects matrix activations/routes and 3D banks")
    if topk_weights.dtype != torch.float32 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("GGUF Q2xQ8 expects FP32 routing weights and integer indices")
    if any(t.device != x.device for t in (gate_up, down, topk_weights, topk_ids)):
        raise ValueError("GGUF Q2xQ8 operands must share one CUDA device")
    if x.dtype != torch.float32 or gate_up.dtype != torch.uint8 or down.dtype != torch.uint8:
        raise ValueError("GGUF Q2×Q8 requires FP32 activation and uint8 expert banks")
    if not all(t.is_contiguous() and t.is_cuda for t in (x, gate_up, down, topk_weights, topk_ids)):
        raise ValueError("GGUF Q2×Q8 requires contiguous CUDA inputs")
    tokens, hidden = x.shape
    intermediate = gate_up.shape[1] // 2
    topk = topk_ids.shape[1]
    routes = tokens * topk
    if (hidden <= 0 or intermediate <= 0 or topk <= 0 or gate_up.shape[0] <= 0
            or gate_up.shape[1] % 2 or hidden % 64 or intermediate % 64
            or gate_up.shape[2] != hidden // 64 * 18
            or down.shape != (gate_up.shape[0], hidden, intermediate // 64 * 18)
            or topk_ids.shape[0] != tokens or topk_weights.shape != topk_ids.shape):
        raise ValueError("incompatible GGUF Q2×Q8 expert/routing geometry")
    if not tokens:
        return torch.empty_like(x)
    gate_values = torch.empty((routes, 2 * intermediate), dtype=torch.float32, device=x.device)
    active = torch.empty((routes, intermediate), dtype=torch.float32, device=x.device)
    routed = torch.empty((routes, hidden), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    # Decode is sparse in route space, independent of total cache capacity.
    # Integer chunk dots and FP32 activation scales match prefill; float reduction
    # trees may differ within the validated bound. Large prefill reuses weights.
    from freetoken.moe.fused import moe_align_block_size
    vector_decode = not is_prefill and tokens <= 8
    if not vector_decode:
        sorted_ids, experts, padded = moe_align_block_size(topk_ids, 16, gate_up.shape[0])
    def project(value, bank, result, input_topk):
        if vector_decode:
            return _routed_project(value, bank, result, topk_ids, input_topk)
        q, scales = quantize_expert_activation(value)
        k, n = value.shape[1], result.shape[1]
        _grouped[(triton.cdiv(sorted_ids.numel(), 16), triton.cdiv(n, 32))](
            q, scales, bank, sorted_ids, experts, padded, result,
            routes, n, k, input_topk, 16, 32, num_warps=4, enable_fp_fusion=False)
    project(x, gate_up, gate_values, topk)
    _grouped_gate_act[(triton.cdiv(routes * intermediate, 256),)](
        gate_values, active, intermediate, routes * intermediate, 256, enable_fp_fusion=False)
    project(active, down, routed, 1)
    _reduce[(tokens, triton.cdiv(hidden, 128))](routed, topk_weights, out, hidden,
        topk, triton.next_power_of_2(topk), 128, num_warps=4, enable_fp_fusion=False)
    return out
