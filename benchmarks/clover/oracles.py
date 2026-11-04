"""Eager batch builders and numerical references; never imported by serving code."""
# Acceptance source: WilliamZR/sglang@2243dc6edb416073b60be0765904cabae9eb61cf
# Path: python/sglang/srt/speculative/clover_utils.py
# Acceptance function SPDX-License-Identifier: Apache-2.0; relocated without arithmetic changes.
import copy

import torch

def clover_correct_drafts(
    logits, candidates, accept_ratio=0.9, accept_delta=0.6, accept_rule="threshold"
):
    """Return counts excluding bonus; logits[j] verifies candidates[j + 1].

    Exact matches the verify argmax; threshold uses raw target probabilities.
    Both are independent of temperature/top-k/top-p used for the boundary token.
    """
    if accept_rule == "exact":
        matches = candidates[:, 1:] == logits[:, :-1].argmax(dim=-1)
        return matches.to(torch.int32).cumprod(dim=-1).sum(-1).to(torch.int32)
    if accept_rule != "threshold":
        raise ValueError(f"Unknown Clover accept rule: {accept_rule}")
    probabilities = torch.softmax(logits[:, :-1].float(), dim=-1)
    proposal_probs = probabilities.gather(-1, candidates[:, 1:, None]).squeeze(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    thresholds = torch.minimum(
        torch.full_like(entropy, accept_ratio), accept_delta * torch.exp(-entropy)
    )
    # A zero threshold must still reject candidates a hard penalty blocked.
    matches = (proposal_probs >= thresholds) & (proposal_probs > 0)
    return matches.to(torch.int32).cumprod(dim=-1).sum(-1).to(torch.int32)


def make_preview_batch(engine, request, tokens, *, coreset_top_p=.9):
    """Build private metadata; no page allocation or live request mutation."""
    from freetoken.attention.linear import FLAMetadata
    from freetoken.core import Batch

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
    batch.clover_kv_splits = 8 if getattr(getattr(engine,'config',None),'clover_split_kv',False) else 1
    batch.clover_valid = torch.ones(width, device=engine.device, dtype=torch.bool)
    batch.clover_coreset_top_p = coreset_top_p
    batch.clover_attention = dict(q_indptr=torch.tensor([0,width], device=engine.device, dtype=torch.int32),
        kv_indptr=torch.tensor([0,prefix+width], device=engine.device, dtype=torch.int32),
        prefix_lens=torch.tensor([prefix], device=engine.device, dtype=torch.int32),
        block_ends=torch.full((width,), prefix+width, device=engine.device, dtype=torch.int32))
    return batch


def make_verify_batch(runtime, request, tokens, *, capacity=16):
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.core import Batch
    from freetoken.models.clover_moe.preview import CloverTransactions

    req = copy.copy(request)
    req.device_len = req.cached_len + tokens.numel()
    req.mamba_ping_pong = None
    batch = Batch([req], "verify")
    batch.padded_reqs = [req]
    batch.input_ids = tokens
    batch.positions = torch.arange(
        req.cached_len, req.device_len, device=runtime.engine.device, dtype=torch.int32
    )
    batch.out_loc = runtime.engine.page_table[
        req.table_idx, req.cached_len : req.device_len
    ]
    batch.linear_table_idx = torch.tensor(
        [req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx],
        device=runtime.engine.device,
        dtype=torch.int32,
    )
    batch.fla_metadata = build_fla_metadata(
        batch, runtime.engine.device, max_verify_rows=len(tokens)
    )
    runtime.engine.attn_backend.prepare_metadata(batch)
    if getattr(runtime.engine.config,'clover_split_kv',False):
        metadata = batch.attn_metadata
        batch.clover_valid = torch.ones(len(tokens),device=runtime.engine.device,dtype=torch.bool)
        batch.clover_kv_splits = 8
        batch.clover_attention = dict(q_indptr=metadata.cu_seqlens_q_gpu,
            kv_indptr=metadata.indptr,indices=metadata.indices,
            prefix_lens=metadata.prefix_lens,block_ends=None)
    if runtime.transactions is None or runtime.transactions.capacity < capacity:
        # Free the old window before allocating a larger one on a 16GB GPU.
        runtime.transactions = None
        runtime.transactions = CloverTransactions(
            runtime.engine.linear_state_pool, capacity
        )
    runtime.transactions.begin(len(tokens))
    batch.clover_transactions = runtime.transactions
    return batch


def predict_with_confidence(logits, gen):
    """Sample with configured filters; confidence always uses T=1 logits."""
    if gen.do_sample:
        scaled = logits / gen.temperature
        if gen.top_k and 0 < gen.top_k < scaled.shape[-1]:
            cutoff = scaled.topk(gen.top_k, -1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < cutoff, -float("inf"))
        if gen.top_p is not None and gen.top_p < 1:
            sorted_logits, order = scaled.sort(-1, descending=True)
            probs = sorted_logits.softmax(-1)
            remove = (probs.cumsum(-1) - probs) > gen.top_p
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            scaled = torch.empty_like(scaled).scatter(-1, order, sorted_logits)
        prediction = torch.multinomial(scaled.softmax(-1), 1).squeeze(-1)
    else:
        prediction = logits.argmax(-1)
    confidence = logits.softmax(-1).gather(-1, prediction[:, None]).squeeze(-1)
    return prediction, confidence


def resolve_low_confidence_reference(preview, state, *, width, mask_id, threshold, device):
    """Official full-width loop, including the final partial returned block.

    `preview` must be read-only. Only the caller may commit the resolved prefix.
    Confidence and selection remain on device; the small step trace is evidence.
    """
    block = torch.full((width,), mask_id, device=device, dtype=torch.int32)
    pending = torch.ones(width, device=device, dtype=torch.bool)
    steps = []
    for _ in range(width):
        logits = preview(block).float().clone()
        logits[:, mask_id] = -float("inf")
        state.apply(logits, block, mask_id)
        prediction, confidence = predict_with_confidence(logits, state.gen)
        confidence = confidence.masked_fill(~pending, -float("inf"))
        transfer = (confidence > threshold) & pending
        if not bool(transfer.any()):
            transfer[confidence.argmax()] = True
        block[transfer] = prediction[transfer].to(block.dtype)
        pending[transfer] = False
        steps.append(int(transfer.sum().item()))
        if not bool(pending.any()):
            return block, steps
    raise RuntimeError("LowConfidence failed to resolve its full block")

