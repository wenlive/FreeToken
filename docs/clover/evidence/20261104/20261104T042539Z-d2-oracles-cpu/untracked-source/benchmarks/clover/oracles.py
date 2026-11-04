"""Eager batch builders and numerical references; never imported by serving code."""
# Acceptance source: WilliamZR/sglang@2243dc6edb416073b60be0765904cabae9eb61cf
# Path: python/sglang/srt/speculative/clover_utils.py
# Acceptance function SPDX-License-Identifier: Apache-2.0; relocated without arithmetic changes.
import copy
import re

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


def iter_hf_gguf_weights(model_path, config, *, preserve_q8=False):
    """Dense tensors in original Clover key/head order for independent HF oracles."""
    from freetoken.models.clover_moe.gguf import _EXPERT, _LAYER, _restore_dense
    from freetoken.models.gguf.reader import iter_gguf_tensors

    import gguf
    import numpy as np

    global_names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "output_norm.weight": "model.norm.weight",
    }
    for tensor in iter_gguf_tensors(model_path):
        if _EXPERT.fullmatch(tensor.name):
            continue
        name = global_names.get(tensor.name)
        if name is None:
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
            if match is None or match[2] not in _LAYER:
                raise ValueError(f"Unmapped Clover GGUF tensor {tensor.name}")
            name = f"model.layers.{int(match[1])}.{_LAYER[match[2]]}"
        # The CPU reader is independent of the CUDA inference dequant kernels.
        raw = tensor._raw
        values = gguf.dequantize(raw, gguf.GGMLQuantizationType(tensor.ggml_type))
        value = torch.from_numpy(np.array(values, copy=True)).reshape(tensor.shape)
        packed_linear = preserve_q8 and tensor.ggml_type == 8 and not name.endswith("embed_tokens.weight")
        dtype = torch.float32 if packed_linear else torch.bfloat16
        yield name, _restore_dense(name, value, config, dtype=dtype)

