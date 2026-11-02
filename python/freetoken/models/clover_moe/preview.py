"""Read-only block preview and causal accepted-prefix transactions.

Preview semantics: DiffuSpace/Clover-35B-A3B-Preview@4b0402b65e523d0183a14c431955438714cb16ff,
modeling_clover_moe.py. Numerical preview kernels retain their SGLang source
notices. Verify reuses FreeToken's existing FLA intermediate-state support.
"""
import copy

import torch

from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch

from freetoken.kernel.causal_conv1d import causal_conv1d_varlen
from freetoken.kernel.triton.attention import extend_paged_attention
from freetoken.kernel.triton.clover_conv import clover_local_memory_conv
from freetoken.kernel.triton.clover_coreset import clover_block_coreset_triton
from freetoken.kernel.triton.clover_gdn import clover_local_memory_triton


class CloverTransactions:
    """One private verify window; live GDN advances only on explicit commit."""
    def __init__(self, pool, max_width):
        if max_width not in (8, 16, 32):
            raise ValueError("Clover transaction width must be 8, 16 or 32")
        self.capacity = max_width
        self.recurrent = torch.empty((pool.recurrent_states.shape[0], max_width, *pool.recurrent_states.shape[2:]),
                                     dtype=pool.recurrent_states.dtype, device=pool.device)
        self.conv = torch.empty((pool.conv_states.shape[0], max_width, *pool.conv_states.shape[2:]),
                                dtype=pool.conv_states.dtype, device=pool.device)
        self.zero = torch.zeros(1, dtype=torch.int32, device=pool.device)
        self.initial = torch.ones(1, dtype=torch.bool, device=pool.device)
        self.width = 0
        self.written = set()

    def begin(self, width):
        if not 1 <= width <= self.capacity:
            raise ValueError("Verify rows exceed the owned transaction capacity")
        self.width = width
        self.written.clear()

    def commit(self, pool, count, destination):
        if not 1 <= count <= self.width or len(self.written) != self.recurrent.shape[0]:
            raise ValueError("Cannot commit an incomplete or out-of-range Clover prefix")
        pool.recurrent_states[:, destination].copy_(self.recurrent[:, count - 1])
        pool.conv_states[:, destination].copy_(self.conv[:, count - 1])

    @property
    def owned_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.recurrent, self.conv, self.zero, self.initial))


def _project(mixer, hidden):
    total = hidden.shape[0]
    if mixer._split_in_proj:
        qkvz = mixer.in_proj_qkvz.forward(hidden)
        raw, z = torch.split(qkvz, [mixer.conv_dim, mixer.value_dim], dim=-1)
        b, a = torch.split(mixer.in_proj_ba.forward(hidden), [mixer.num_v_heads] * 2, dim=-1)
    else:
        raw, z, b, a = torch.split(mixer.in_proj.forward(hidden), mixer._in_proj_split, dim=-1)
    return raw, z.reshape(total, mixer.num_v_heads, mixer.head_v_dim), b, a


def gdn_forward(mixer, hidden, ctx, *, preview):
    batch, pool = ctx.batch, ctx.linear_state_pool
    if len(batch.reqs) != 1:
        raise ValueError("Clover block forwards currently require one request")
    raw, z, b, a = _project(mixer, hidden)
    total, dtype = hidden.shape[0], hidden.dtype
    indices = batch.fla_metadata.cache_indices
    li = pool.local_index(mixer.layer_id)
    if preview:
        valid = batch.clover_valid
        mixed = clover_local_memory_conv(raw, clean_conv_states=pool.conv_states[li], cache_indices=indices,
            conv_weights=mixer._conv_weight(), bias=None, activation="silu", block_size=total,
            active_valid_mask=valid, conv_mode="independent_symmetric", conv_right_weight=mixer.conv_right_weight)
    else:
        transaction = batch.clover_transactions
        if transaction.width != total:
            raise ValueError("Clover verify input does not match the active transaction")
        slot = batch.reqs[0].linear_slot_idx
        if slot is None:
            slot = batch.reqs[0].table_idx
        clean = pool.conv_states[li, slot:slot + 1]
        # Save raw-input windows before a causal-convolution implementation is
        # allowed to overwrite its input. Scratch history never aliases live.
        joined = torch.cat((clean[0].T, raw), dim=0)
        transaction.conv[li, :total].copy_(joined.unfold(0, mixer.conv_kernel_size - 1, 1)[1:total + 1])
        mixed = causal_conv1d_varlen(raw.T.contiguous().clone(), mixer._conv_weight(), clean.clone(),
            batch.fla_metadata.cu_seqlens, transaction.zero, transaction.initial).T.contiguous()
    qf, kf, vf = torch.split(mixed, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1)
    q = qf.reshape(1, total, mixer.num_k_heads, mixer.head_k_dim).to(dtype)
    k = kf.reshape(1, total, mixer.num_k_heads, mixer.head_k_dim).to(dtype)
    v = vf.reshape(1, total, mixer.num_v_heads, mixer.head_v_dim).to(dtype)
    if preview:
        g, _ = mixer._gate_params(a, b)
        output = clover_local_memory_triton(q, k, v, g.reshape(1, total, mixer.num_v_heads),
            clean_ssm_states=pool.recurrent_states[li], cache_indices=indices, active_valid_mask=valid,
            scale=mixer.head_k_dim ** -.5, softmax_logit_scale=mixer.bd_softmax_logit_scale,
            softmax_history_bias=mixer.bd_softmax_history_bias, softmax_history_c=mixer.bd_softmax_history_c)
    else:
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        output = fused_sigmoid_gating_delta_rule_update(A_log=mixer.A_log, a=a, dt_bias=mixer.dt_bias,
            softplus_beta=1., softplus_threshold=20., q=q, k=k, v=v, b=b,
            initial_state_source=pool.recurrent_states[li], initial_state_indices=indices,
            scale=mixer.head_k_dim ** -.5, use_qk_l2norm_in_kernel=True,
            cu_seqlens=batch.fla_metadata.cu_seqlens, disable_state_update=True,
            intermediate_states_buffer=transaction.recurrent[li].unsqueeze(0),
            intermediate_state_indices=transaction.zero)
        transaction.written.add(li)
    normalized = mixer.norm.forward(output.reshape(-1, mixer.head_v_dim), z.reshape(-1, mixer.head_v_dim))
    return mixer.out_proj.forward(normalized.reshape(total, -1))


def attention_preview(mixer, hidden, ctx):
    batch = ctx.batch
    req = batch.reqs[0]
    q, k, v, gate = mixer._project(hidden)
    width, prefix = hidden.shape[0], req.cached_len
    kh, dim = mixer.num_kv, mixer.head_dim
    keys = ctx.kv_cache.k_cache(mixer.layer_id).view(-1, kh, dim)
    values = ctx.kv_cache.v_cache(mixer.layer_id).view(-1, kh, dim)
    metadata = batch.clover_attention
    output = extend_paged_attention(q, keys, values, metadata["q_indptr"], metadata["kv_indptr"],
        ctx.page_table[req.table_idx, :prefix], metadata["prefix_lens"], width, dim ** -.5,
        k_extend=k.reshape(width, kh, dim), v_extend=v.reshape(width, kh, dim), block_ends=metadata["block_ends"])
    return mixer._combine(output, gate)


def moe_preview(mixer, hidden, ctx):
    router = mixer.gate.forward(hidden)
    if ctx.batch.clover_coreset_top_p < 1.:
        router = clover_block_coreset_triton(router, hidden.shape[0], mixer.experts.top_k,
                                            ctx.batch.clover_coreset_top_p)
    # The routed kernel can overwrite its input, so shared projections go first.
    shared = mixer.shared_expert.forward(hidden) * torch.sigmoid(mixer.shared_expert_gate.forward(hidden))
    return mixer.experts.forward(hidden_states=hidden, router_logits=router) + shared


def block_forward(model, ctx, *, preview):
    batch = ctx.batch
    hidden = model.model.embed_tokens.forward(batch.input_ids)
    residual = None
    for layer in model.model.layers.op_list:
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm.forward(hidden)
        else:
            hidden, residual = layer.input_layernorm.forward_add_residual(hidden, residual)
        if layer._is_linear:
            hidden = gdn_forward(layer.linear_attn, hidden, ctx, preview=preview)
        elif preview:
            hidden = attention_preview(layer.self_attn, hidden, ctx)
        else:
            hidden = layer.self_attn.forward(hidden)
        hidden, residual = layer.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = moe_preview(layer.mlp, hidden, ctx) if preview else layer.mlp.forward(hidden)
    hidden, _ = model.model.norm.forward_add_residual(hidden, residual)
    return model.lm_head.forward(hidden)


def make_preview_batch(engine, request, tokens, *, coreset_top_p=.9):
    """Build private metadata; no page allocation or live request mutation."""
    width = len(tokens)
    if width not in (8, 16, 32) or not 0 < coreset_top_p <= 1:
        raise ValueError("Clover preview needs width 8/16/32 and coreset top-p in (0,1]")
    prefix = request.cached_len
    if prefix + width > engine.max_seq_len:
        raise ValueError("Clover preview block exceeds the engine's context capacity")
    req = copy.copy(request)
    req.device_len = prefix + width
    req.mamba_ping_pong = None
    batch = Batch([req], "preview")
    batch.padded_reqs = [req]
    batch.input_ids = torch.as_tensor(tokens, device=engine.device, dtype=torch.int32)
    batch.positions = torch.arange(prefix, prefix + width, device=engine.device, dtype=torch.int32)
    batch.out_loc = None
    slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
    batch.linear_table_idx = torch.tensor([slot], device=engine.device, dtype=torch.int32)
    batch.fla_metadata = FLAMetadata(torch.tensor([0,width], device=engine.device, dtype=torch.int32),
        batch.linear_table_idx, torch.ones(1, device=engine.device, dtype=torch.bool))
    batch.clover_preview = True
    batch.clover_valid = torch.ones(width, device=engine.device, dtype=torch.bool)
    batch.clover_coreset_top_p = coreset_top_p
    batch.clover_attention = dict(q_indptr=torch.tensor([0,width], device=engine.device, dtype=torch.int32),
        kv_indptr=torch.tensor([0,prefix+width], device=engine.device, dtype=torch.int32),
        prefix_lens=torch.tensor([prefix], device=engine.device, dtype=torch.int32),
        block_ends=torch.full((width,), prefix+width, device=engine.device, dtype=torch.int32))
    return batch
