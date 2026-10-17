"""One-request greedy MTP windows with owned transactions and native scheduler delivery."""
from __future__ import annotations

import copy
from collections import deque
from typing import NamedTuple

import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch
from .mtp_draft_graph import DraftGraphs
from .mtp_state import MTP_LAST_RESIDUAL


class DeliveryLimit(NamedTuple):
    count: int
    finished: bool


def verified_prefix(candidates, picks, eos_ids, ignore_eos):
    accepted = 0
    for draft, target in zip(candidates, picks):
        if draft != target or (not ignore_eos and draft in eos_ids):
            break
        accepted += 1
    return [*candidates[:accepted], picks[accepted]]


def window_depth(req, manager, max_context):
    """Speculation uses owned tail pages or currently reclaimable pages only."""
    capacity = req.mtp_reserved_len + manager.available_size
    return max(0, min(3, req.remain_len - 1, max_context - req.cached_len,
                      capacity - req.device_len))


class MtpRuntime:
    def __init__(self, engine, drafter):
        self.engine = engine
        self.draft = drafter
        self.ple = engine.model._ple_table
        self.graphs = {}
        self.draft_graphs = None
        self.active_owner = None
        # Diagnostics must stay bounded in a long-lived Speed worker.
        self.rounds = deque(maxlen=4096)
        self.prepare_graphs()

    @property
    def pool(self):
        return self.engine.linear_state_pool

    @property
    def aux(self):
        return self.engine.kv_cache.mtp_state

    @property
    def raw(self):
        return self.engine.model.model._mtp_residual_capture

    def eligible(self, batch):
        if not batch.is_decode or len(batch.reqs) != 1:
            return False
        req = batch.reqs[0]
        return (req.sampling_params.is_greedy and not req.mm_items and req.extend_len == 1
                and 0 < req.cached_len < self.draft.max_context and req.remain_len > 0)

    def prepare_graphs(self):
        if len(self.pool.reserved_slot_ids) != 4:
            raise ValueError('MTP needs four exclusively reserved transaction slots')
        self.dst = torch.tensor(self.pool.reserved_slot_ids, device=self.engine.device, dtype=torch.int64)
        self.rings = torch.empty((4, *self.engine.kv_cache._pending_ring.shape[1:]),
                                 device=self.engine.device, dtype=self.engine.kv_cache._pending_ring.dtype)
        for count in (1, 2, 3, 4):
            req = copy.copy(self.engine.dummy_req)
            req.cached_len, req.device_len = 1, 1 + count
            req.input_ids = torch.zeros(1 + count, dtype=torch.int32, device='cpu')
            req.mamba_ping_pong = None
            batch = Batch([req], 'verify')
            batch.padded_reqs = [req]
            batch.input_ids = torch.zeros(count, device=self.engine.device, dtype=torch.int32)
            batch.positions = torch.arange(1, count + 1, device=self.engine.device, dtype=torch.int32)
            dummy_base = self.engine.num_pages * self.engine.config.page_size
            batch.out_loc = dummy_base + batch.positions
            batch.linear_table_idx = torch.tensor([0], device=self.engine.device, dtype=torch.int32)
            batch.fla_metadata = build_fla_metadata(batch, self.engine.device)
            fla = batch.fla_metadata
            if self.pool.compact_mtp:
                fla.compact_mtp_pool = self.pool
            fla.track_dst = self.dst[:count]
            fla.track_boundary_row = torch.arange(1, count + 1, device=self.engine.device)
            batch.mtp_ring_snapshots = self.rings
            self.engine.attn_backend.prepare_metadata(batch)
            for _ in range(2):
                with self.engine.ctx.forward_batch(batch), self.engine.model.forward_host_ctx(batch, False):
                    hidden = self.engine.model.model.forward(batch.input_ids, batch)
                    self.engine.model.lm_head.quant_method.apply(self.engine.model.lm_head, hidden)
            torch.cuda.synchronize(self.engine.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.engine.stream):
                with self.engine.ctx.forward_batch(batch):
                    hidden = self.engine.model.model.forward(batch.input_ids, batch)
                    logits = self.engine.model.lm_head.quant_method.apply(self.engine.model.lm_head, hidden)
                    picks = logits.argmax(-1).to(torch.int32)
            self.graphs[count] = (batch, graph, picks)
        self.draft_graphs = DraftGraphs(self.draft, self.engine.stream, attention_width=self.draft.max_context)
        self.pool.clear_slots([self.pool.padding_slot, *self.pool.reserved_slot_ids])
        self.engine.kv_cache._pending_ring[self.engine.dummy_req.table_idx].zero_()
        self.active_owner = None

    def close_graphs(self):
        torch.cuda.synchronize(self.engine.device)
        for _, graph, _ in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        if self.draft_graphs is not None:
            self.draft_graphs.close()
            self.draft_graphs = None
        self.active_owner = None

    def _ensure_owner(self, req):
        if self.active_owner is not req:
            if req.cached_len <= self.draft.max_context:
                self.aux.restore_prefix(self.engine.page_table[req.table_idx, :req.cached_len],
                                        self.draft.keys, self.draft.values)
            self.active_owner = req

    def before_native(self, batch):
        req = batch.reqs[0]
        self._ensure_owner(req)
        previous = None
        if 0 < req.cached_len <= self.draft.max_context:
            previous = self.pool.slot_state(MTP_LAST_RESIDUAL)[req.linear_slot_idx:req.linear_slot_idx + 1].clone()
        return req.cached_len, previous

    def after_native(self, batch, native_state):
        base, previous = native_state
        req = batch.reqs[0]
        if previous is not None:
            self.draft.prime(previous, batch.input_ids[:1], start=base - 1)
            self.aux.publish(batch.out_loc[:1], self.draft.keys[base-1:base], self.draft.values[base-1:base])
        count = min(len(batch.input_ids) - 1, self.draft.max_context - base)
        if count > 0:
            self.draft.prime(self.raw[:count], batch.input_ids[1:count + 1], start=base)
            self.aux.publish(batch.out_loc[1:count + 1], self.draft.keys[base:base + count],
                             self.draft.values[base:base + count])

    def _replay(self, req, inputs, position):
        batch, graph, picks = self.graphs[len(inputs)]
        batch.input_ids.copy_(torch.tensor(inputs, device=self.engine.device, dtype=torch.int32))
        batch.positions.copy_(torch.arange(position, position + len(inputs), device=self.engine.device, dtype=torch.int32))
        batch.out_loc.copy_(self.engine.page_table[req.table_idx, batch.positions.long()])
        batch.fla_metadata.cache_indices.fill_(req.linear_slot_idx)
        batch.linear_table_idx.fill_(req.linear_slot_idx)
        md = batch.attn_metadata
        md.ring_slots.fill_(req.table_idx)
        md.seq_lens.fill_(position + len(inputs))
        md.kv_len_cpu.fill_(position + len(inputs))
        md.block_table.copy_(self.engine.attn_backend._block_table(md.ring_slots.long()))
        context = self.ple._ple_context(req.input_ids, position)
        self.ple.fill([torch.tensor([*context, *inputs], dtype=torch.int64)], graph=True)
        graph.replay()
        self.ple._graph_consumed.record(self.engine.stream)
        return picks

    def _freeze_anchor(self, req, position, committed, delivered, limit, anchor_id):
        if anchor_id is None or req.mamba_last_track_seqlen is not None:
            return
        anchor = req.toolcall_anchor_len
        if anchor is None:
            for i, token in enumerate(delivered):
                if token == anchor_id and not (i == len(delivered) - 1 and limit.finished):
                    anchor = position + i + 2
                    req.toolcall_anchor_len = anchor
                    break
        if (anchor is not None and position < anchor <= position + committed
                and anchor % self.engine.config.page_size == 0):
            row = anchor - position - 1
            dst = req.mamba_ping_pong[req.mamba_next_track_idx]
            self.pool.copy_from(self.pool.reserved_slot_ids[row], dst)
            req.mamba_last_track_seqlen = anchor
            req.mamba_next_track_idx = 1 - req.mamba_next_track_idx

    def forward(self, batch, manager, token_pool, eos_ids, clip, anchor_id):
        if self.engine.prefill_workspace is not None:
            self.engine.prefill_workspace.check_usable()
        from freetoken.engine.engine import ForwardOutput
        if self.engine.q2_adaptive_owners is not None:
            self.engine.q2_adaptive_owners.step()
        req = batch.reqs[0]
        self._ensure_owner(req)
        position = req.cached_len
        depth = window_depth(req, manager, self.draft.max_context)
        manager.allocate_paged([req], reserve_tokens=depth)
        pending = int(batch.input_ids[0].item())
        residual = self.pool.slot_state(MTP_LAST_RESIDUAL)[req.linear_slot_idx:req.linear_slot_idx + 1]
        if depth:
            candidates = self.draft_graphs.chain(residual, pending, position - 1, depth)
        else:
            candidates = []
            self.draft_graphs.prime(residual, batch.input_ids[:1], position - 1)
        self.aux.publish(self.engine.page_table[req.table_idx, position:position + 1],
                         self.draft.keys[position-1:position], self.draft.values[position-1:position])
        inputs = [pending, *candidates]
        picks = self._replay(req, inputs, position).cpu().tolist()
        delivered = verified_prefix(candidates, picks, eos_ids, req.sampling_params.ignore_eos)
        limit = clip(req, delivered)
        delivered = delivered[:limit.count]
        committed = len(delivered)
        if self.pool.compact_mtp:
            # The compact transaction still reads live as its base: freeze an anchor before advancing live.
            self._freeze_anchor(req, position, committed, delivered, limit, anchor_id)
            self.pool.copy_from(self.pool.reserved_slot_ids[committed - 1], req.linear_slot_idx)
            self.engine.kv_cache._pending_ring[req.table_idx].copy_(self.rings[committed - 1])
        elif committed < len(inputs):
            self.pool.copy_from(self.pool.reserved_slot_ids[committed - 1], req.linear_slot_idx)
            self.engine.kv_cache._pending_ring[req.table_idx].copy_(self.rings[committed - 1])
        if not self.pool.compact_mtp:
            self._freeze_anchor(req, position, committed, delivered, limit, anchor_id)
        accepted = committed - 1
        tokens = torch.tensor(delivered, device=self.engine.device, dtype=torch.int32)
        if accepted:
            self.draft_graphs.prime(self.raw[:accepted], tokens[:accepted], position)
            self.aux.publish(self.engine.page_table[req.table_idx, position+1:position+committed],
                             self.draft.keys[position:position+accepted], self.draft.values[position:position+accepted])
        old_device = req.device_len
        req.cached_len = position + committed
        req.device_len = old_device + committed
        token_pool[req.table_idx, old_device:req.device_len].copy_(tokens)
        bundle = tokens.to('cpu', non_blocking=True)
        event = torch.cuda.Event()
        event.record(self.engine.stream)
        batch.mtp_token_ids = bundle
        self.rounds.append(dict(uid=req.uid, position=position, draft_count=depth,
                               accepted_drafts=accepted, committed_rows=committed))
        return ForwardOutput(tokens[-1:], bundle[-1:], event)
