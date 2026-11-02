"""Single-request Clover preview/verify graphs with device-side expert offload.

Uses FreeToken's existing LRU and mapped-host copy kernels unchanged. Both
backends consume the same stable metadata and a private clean GDN prefix.
"""

import copy
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from freetoken.attention.linear import FLAMetadata
from freetoken.attention.triton import TritonMetadata
from freetoken.core import Batch


@triton.jit
def _stage(packet, table, positions, out_loc, indices, prefix_lens, kv_indptr,
           block_ends, WIDTH: tl.constexpr, MAX_SEQ: tl.constexpr,
           STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    prefix = tl.load(packet)
    owner = tl.load(packet + 1).to(tl.int64)
    tl.store(positions + col, prefix + col, col < WIDTH)
    location = tl.load(table + owner * STRIDE + prefix + col, col < WIDTH, 0)
    tl.store(out_loc + col, location, col < WIDTH)
    slot = tl.load(table + owner * STRIDE + col, col < MAX_SEQ, 0)
    tl.store(indices + col, slot, col < MAX_SEQ)
    tl.store(block_ends + col, prefix + WIDTH, col < WIDTH)
    if tl.program_id(0) == 0:
        tl.store(prefix_lens, prefix)
        tl.store(kv_indptr + 1, prefix + WIDTH)


class CloverBlockGraphs:
    def __init__(self, engine):
        self.engine = engine
        self.entries = {}
        self.clean = None
        self.clean_owner = None
        self.clean_position = None
        self.packet_host = torch.empty(2, dtype=torch.int32, pin_memory=True)
        self.packet = torch.empty(2, dtype=torch.int32, device=engine.device)
        self.pool = None

    def _allocate_clean(self):
        if self.clean is None:
            pool = self.engine.linear_state_pool
            self.clean = SimpleNamespace(
                recurrent_states=torch.empty_like(pool.recurrent_states[:, :1]),
                conv_states=torch.empty_like(pool.conv_states[:, :1]),
            )

    def prepare_clean(self, req, *, allow_ahead=False):
        self._allocate_clean()
        if allow_ahead and self.clean_owner is req and self.clean_position == req.device_len:
            return True
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        pool = self.engine.linear_state_pool
        self.clean.recurrent_states.copy_(pool.recurrent_states[:, slot:slot + 1])
        self.clean.conv_states.copy_(pool.conv_states[:, slot:slot + 1])
        self.clean_owner = None
        self.clean_position = None
        return False

    def save_ahead(self, req, transaction, count):
        self.clean.recurrent_states[:, 0].copy_(transaction.recurrent[:, count - 1])
        self.clean.conv_states[:, 0].copy_(transaction.conv[:, count - 1])
        self.clean_owner = req
        self.clean_position = req.device_len

    def restore_clean(self, req):
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        pool = self.engine.linear_state_pool
        pool.recurrent_states[:, slot:slot + 1].copy_(self.clean.recurrent_states)
        pool.conv_states[:, slot:slot + 1].copy_(self.clean.conv_states)

    def _batch(self, request, rows, preview, coreset, transaction):
        engine = self.engine
        self._allocate_clean()
        req = copy.copy(request)
        req.mamba_ping_pong = None
        req.device_len = req.cached_len + rows
        batch = Batch([req], "preview" if preview else "verify")
        batch.padded_reqs = [req]
        batch.input_ids = torch.zeros(rows, dtype=torch.int32, device=engine.device)
        batch.positions = torch.empty_like(batch.input_ids)
        batch.out_loc = torch.empty_like(batch.input_ids)
        batch.linear_table_idx = torch.zeros(1, dtype=torch.int32, device=engine.device)
        cu = torch.tensor([0, rows], dtype=torch.int32, device=engine.device)
        prefix = torch.empty(1, dtype=torch.int32, device=engine.device)
        indptr = torch.zeros(2, dtype=torch.int32, device=engine.device)
        indices = torch.empty(engine.max_seq_len, dtype=torch.int32, device=engine.device)
        ends = torch.empty(rows, dtype=torch.int32, device=engine.device)
        batch.fla_metadata = FLAMetadata(cu, batch.linear_table_idx,
            torch.ones(1, dtype=torch.bool, device=engine.device), is_verify=not preview)
        batch.clover_clean = self.clean
        batch.clover_preview = preview
        batch.clover_coreset_top_p = coreset
        batch.clover_valid = torch.ones(rows, dtype=torch.bool, device=engine.device)
        batch.clover_attention = dict(q_indptr=cu, kv_indptr=indptr, indices=indices,
                                     prefix_lens=prefix, block_ends=ends)
        batch.attn_metadata = TritonMetadata(cu_seqlens_q_gpu=cu, indptr=indptr,
            indices=indices, q_to_req=torch.zeros(rows, dtype=torch.int32, device=engine.device),
            q_positions=batch.positions, is_decode=False, prefix_lens=prefix, max_q_len=rows)
        batch.clover_transactions = transaction
        return batch

    def stage(self, batch):
        meta = batch.clover_attention
        _stage[(triton.cdiv(self.engine.max_seq_len, 256),)](
            self.packet, self.engine.page_table, batch.positions, batch.out_loc,
            meta["indices"], meta["prefix_lens"], meta["kv_indptr"], meta["block_ends"],
            len(batch.input_ids), self.engine.max_seq_len, self.engine.page_table.stride(0), 256)

    def run(self, req, tokens, *, preview, coreset=.9, transaction=None):
        rows = len(tokens)
        key = (preview, rows, coreset if preview else 1.)
        self.packet_host[0] = req.cached_len
        self.packet_host[1] = req.table_idx
        self.packet.copy_(self.packet_host, non_blocking=True)
        if not preview:
            transaction.begin(rows)
        if key not in self.entries:
            batch = self._batch(req, rows, preview, coreset, transaction)
            batch.input_ids.copy_(tokens)
            self.stage(batch)
            with self.engine.ctx.forward_batch(batch):
                for _ in range(2):
                    self.engine.model.forward()
            torch.cuda.synchronize(self.engine.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, stream=self.engine.stream):
                self.stage(batch)
                with self.engine.ctx.forward_batch(batch):
                    output = self.engine.model.forward()
            if self.pool is None:
                self.pool = graph.pool()
            self.entries[key] = (batch, graph, output)
        batch, graph, output = self.entries[key]
        batch.input_ids.copy_(tokens)
        if not preview:
            transaction.written.update(range(transaction.recurrent.shape[0]))
        graph.replay()
        return output

    def clear_verify(self):
        self.entries = {key: entry for key, entry in self.entries.items() if key[0]}

    def close(self):
        self.entries.clear()
        self.pool = None
        self.clean = None
        self.clean_owner = None
