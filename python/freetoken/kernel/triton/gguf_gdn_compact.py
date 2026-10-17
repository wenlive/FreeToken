"""Compact FP32 GDN transactions: read-only live base and accepted-prefix replay."""
import torch
import triton
import triton.language as tl

@triton.jit
def _verify(Q, K, V, A, B, NEG_A, DT, STATE, CU, SLOTS, OUT,
            KHIST, DHIST, DECAYS, SOURCE, HK: tl.constexpr, HV: tl.constexpr,
            D: tl.constexpr, QS: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
            KS: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
            VS: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
            AS: tl.constexpr, BS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    begin, end = tl.load(CU + req), tl.load(CU + req + 1)
    slot = tl.load(SLOTS + req).to(tl.int64)
    if head == 0:
        tl.store(SOURCE, slot)
    offset = tl.arange(0, D)
    state_offset = (slot * HV + head) * D * D + offset[:, None] * D + offset[None, :]
    state = tl.load(STATE + state_offset).to(tl.float32)
    negative_a, dt = tl.load(NEG_A + head), tl.load(DT + head)
    q_head = head % HK
    for row in range(begin, end):
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
        history_offset = ((row - begin) * HV + head) * D + offset
        tl.store(KHIST + history_offset, k)
        tl.store(DHIST + history_offset, delta)
        tl.store(DECAYS + (row - begin) * HV + head, decay)


@triton.jit
def _commit(STATE, KHIST, DHIST, DECAYS, SOURCE, DST: tl.constexpr,
            SLOTS: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, COUNT: tl.constexpr):
    layer, head = tl.program_id(0), tl.program_id(1)
    source = tl.load(SOURCE + layer).to(tl.int64)
    offset = tl.arange(0, D)
    base = offset[:, None] * D + offset[None, :]
    state = tl.load(STATE + ((layer * SLOTS + source) * HV + head) * D * D + base)
    for row in range(COUNT):
        history_offset = ((layer * 4 + row) * HV + head) * D + offset
        k = tl.load(KHIST + history_offset)
        delta = tl.load(DHIST + history_offset)
        decay = tl.load(DECAYS + (layer * 4 + row) * HV + head)
        state = decay * state + k[:, None] * delta[None, :]
    tl.store(STATE + ((layer * SLOTS + DST) * HV + head) * D * D + base, state)



def verify(q,k,v,a,b,negative_a,dt_bias,state,fla,output,*,is_decode):
    """Keep live unchanged and record the four candidate recurrence updates."""
    pool=fla.compact_mtp_pool
    if is_decode or not fla.is_verify or fla.cache_indices.numel()!=1 or not 1<=len(q)<=4:
        raise ValueError('Compact verification requires one request and1-4 rows')
    layer=state.storage_offset()//state.numel()
    _,hk,dim=q.shape
    heads=v.shape[1]
    _verify[(1,heads)](q,k,v,a,b,negative_a,dt_bias,state,fla.cu_seqlens,fla.cache_indices,output,
        pool._compact_keys[layer],pool._compact_delta[layer],pool._compact_decay[layer],
        pool._compact_source[layer:layer+1],hk,heads,dim,*q.stride(),*k.stride(),*v.stride(),
        a.stride(0),b.stride(0),num_warps=8,enable_fp_fusion=False)
    return output


def commit_prefix(pool,count,dst):
    """Replay only accepted updates across all layers in one launch, without inverse arithmetic."""
    layers,slots,heads,dim,_=pool.recurrent_states.shape
    _commit[(layers,heads)](pool.recurrent_states,pool._compact_keys,pool._compact_delta,
        pool._compact_decay,pool._compact_source,dst,slots,heads,dim,count,
        num_warps=8,enable_fp_fusion=False)
