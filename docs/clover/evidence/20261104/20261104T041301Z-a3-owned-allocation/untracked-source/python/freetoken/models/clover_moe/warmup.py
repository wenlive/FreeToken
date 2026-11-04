"""Capture declared block shapes using temporarily owned cache resources."""

import time
from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from .preview import CloverTransactions


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
    manager.ensure_mamba_slots(1)
    slot = pool.alloc(1)[0]
    table = tables.allocate()
    row = engine.page_table[table].clone()
    handle = manager.match_req(SimpleNamespace(input_ids=[0], input_len=1), max_len=0).cuda_handle
    manager.lock(handle)
    req = Req(torch.zeros(1, dtype=torch.int32), table, 0, maximum, -1,
        SamplingParams(clover_mode="causal"), handle, linear_slot_idx=slot)
    state_copies, kv_copies = [], []
    timings = []
    owns_pages = False
    try:
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
            runtime._preview(req, tokens)
            runtime._verify(req, tokens, width)
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
            manager.unlock(handle)
            pool.free([slot])
        engine.page_table[table].copy_(row)
        tables.free(table)
        if engine.moe_offload_cache is not None:
            engine.moe_offload_cache.reset()
        torch.cuda.synchronize(engine.device)
    return dict(widths=list(widths), coreset_top_p=engine.config.clover_coreset_top_p,
        seconds=time.perf_counter() - started, captures=timings,
        allocated_bytes_delta=torch.cuda.memory_allocated(engine.device) - allocated,
        reserved_bytes_delta=torch.cuda.memory_reserved(engine.device) - reserved)
