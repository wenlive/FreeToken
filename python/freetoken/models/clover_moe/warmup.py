"""Capture declared block shapes using temporarily owned cache resources."""

import time
from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from .preview import CloverTransactions


def _warmup_sampling(runtime, preview, scores, tokens):
    """Compile the declared penalty/pick paths without committing request history."""
    from .generation import CloverPenaltyState
    from freetoken.kernel.triton.clover_pick import pick_with_confidence, resolve_step
    from freetoken.kernel.triton.clover_verify import penalize_logits, verify_and_deliver

    config = runtime.engine.config
    state = CloverPenaltyState(runtime.generation, scores.shape[-1], scores.device, runtime.stop_ids)
    picks, confidence = pick_with_confidence(preview, state, tokens, runtime.mask_id)
    resolve_step(tokens.clone(), torch.ones_like(tokens, dtype=torch.bool), picks, confidence,
        config.clover_threshold, torch.empty(2, dtype=torch.int32, device=tokens.device))
    penalize_logits(scores[:1], state, runtime.mask_id)
    if tokens.numel() in (8, 16):
        pick_with_confidence(preview[1:], state, tokens[1:], runtime.mask_id)
        verify_and_deliver(scores, tokens, state, runtime.mask_id,
            rule=config.clover_accept_rule, ratio=config.clover_accept_ratio,
            delta=config.clover_accept_delta)


@torch.inference_mode()
def warmup_block_graphs(runtime, manager, tables):
    engine = runtime.engine
    runtime._require_block_backend(manager)
    widths = engine.config.clover_graph_warmup_widths
    maximum = max(widths)
    if not manager.is_hybrid or maximum > engine.max_seq_len:
        raise ValueError("Clover graph warmup requires hybrid cache and room for its largest width")
    if not tables.available_size:
        raise ValueError("Clover graph warmup requires an idle request table")
    started = time.perf_counter()
    allocated = torch.cuda.memory_allocated(engine.device)
    reserved = torch.cuda.memory_reserved(engine.device)
    pool = engine.linear_state_pool
    slot = table = row = handle = None
    state_copies, kv_copies = [], []
    timings = []
    owns_pages = False
    try:
        manager.ensure_mamba_slots(1)
        slot = pool.alloc(1)[0]
        table = tables.allocate()
        row = engine.page_table[table].clone()
        candidate = manager.match_req(SimpleNamespace(input_ids=[0], input_len=1), max_len=0).cuda_handle
        manager.lock(candidate)
        handle = candidate
        req = Req(torch.zeros(1, dtype=torch.int32), table, 0, maximum, -1,
            SamplingParams(clover_mode="causal"), handle, linear_slot_idx=slot)
        manager.allocate_paged([req], reserve_tokens=maximum - 1)
        owns_pages = True
        locations = engine.page_table[table, :maximum].long()
        for tensor in (pool.conv_states[:, slot], pool.recurrent_states[:, slot]):
            state_copies.append((tensor, tensor.cpu()))
            tensor.zero_()
        for layer in engine.model.model.layers.op_list:
            if layer._is_linear:
                continue
            for cache in (engine.kv_cache.k_cache(layer.self_attn.layer_id),
                          engine.kv_cache.v_cache(layer.self_attn.layer_id)):
                flat = cache.view(-1, *cache.shape[-2:])
                kv_copies.append((flat, flat[locations].cpu()))
        blocks = runtime._blocks()
        # Every verify graph captures the same storage, including the smaller widths.
        if runtime.transactions is None or runtime.transactions.capacity < maximum:
            blocks.clear_verify()
            runtime.transactions = CloverTransactions(pool, maximum)
        blocks.prepare_clean(req)
        for width in widths:
            tokens = torch.full((width,), runtime.mask_id, dtype=torch.int32, device=engine.device)
            begin = time.perf_counter()
            preview = runtime._preview(req, tokens)
            scores = runtime._verify(req, tokens, width)
            _warmup_sampling(runtime, preview, scores, tokens)
            del preview, scores
            torch.cuda.synchronize(engine.device)
            timings.append(dict(width=width, seconds=time.perf_counter() - begin))
    finally:
        # Verification writes only these owned KV positions; never publish a synthetic prefix.
        for flat, backup in kv_copies:
            flat.index_copy_(0, locations, backup.to(engine.device))
        for tensor, backup in state_copies:
            tensor.copy_(backup)
        torch.cuda.synchronize(engine.device)
        if owns_pages:
            manager.cache_req(req, finished=True)
        else:
            if handle is not None:
                manager.unlock(handle)
            if slot is not None:
                pool.free([slot])
        if row is not None:
            engine.page_table[table].copy_(row)
        if table is not None:
            tables.free(table)
        if engine.moe_offload_cache is not None:
            engine.moe_offload_cache.reset()
        torch.cuda.synchronize(engine.device)
    return dict(widths=list(widths), coreset_top_p=engine.config.clover_coreset_top_p,
        seconds=time.perf_counter() - started, captures=timings,
        allocated_bytes_delta=torch.cuda.memory_allocated(engine.device) - allocated,
        reserved_bytes_delta=torch.cuda.memory_reserved(engine.device) - reserved)
