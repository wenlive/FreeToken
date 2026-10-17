"""Compressed Q2 experts with disjoint protected, dynamic and prefill regions.

Owners are GPU rows or compact pinned host rows. Dynamic cache entries
are disposable copies. Keeping those roles separate prevents a GPU-only owner
from being evicted or overwritten by prefill, including across graph replay.
Opt-in online exchange changes the owner set only after all users drain.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from freetoken.utils import init_logger
from .offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass(frozen=True)
class Q2Ownership:
    sources: dict
    caches: dict
    host_indices: list
    protected_slots: dict
    initial_map: torch.Tensor
    slots: int
    dynamic: int
    protected_begin: int
    layers: int
    experts: int
    layout: dict
    host_bytes: int
    gpu_bank_bytes: int


def protected_ranking(layers, experts, count):
    """Use the frozen initial ranking; online adaptation may later observe completed routes only."""
    path = Path(__file__).parent / 'configs/q2_residency/strata_flash_next.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() != '171adfd563ddf441deb908fd9432b2411bed435ce3e640c54168aae2d88bbfd5':
        raise ValueError('Packaged Q2 expert ranking has changed')
    profile = json.loads(raw)
    if (profile['format'], profile['layers'], profile['experts']) != (1, layers, experts):
        raise ValueError('Protected Q2 ranking does not match this model geometry')
    ranked = [tuple(pair) for pair in profile['ranked']]
    if not 0 < count <= len(ranked):
        raise ValueError('Protected Q2 count exceeds the frozen expert ranking')
    return ranked[:count]


@triton.jit
def _owner_gather(DST, SOURCES, FEATS, DI, SI, COUNT,
                  TABLE_SIZE: tl.constexpr, BLOCKS: tl.constexpr,
                  UNITS: tl.constexpr):
    bank = tl.program_id(0)
    block = tl.program_id(1)
    count = tl.load(COUNT)
    units_per_row = tl.load(FEATS + bank) // 16
    destination = tl.load(DST + bank)
    total = count * units_per_row
    lanes = tl.arange(0, UNITS)
    words = tl.arange(0, 4)
    for start in range(0, tl.cdiv(total, BLOCKS * UNITS)):
        unit = start * (BLOCKS * UNITS) + block * UNITS + lanes
        valid = unit < total
        route = unit // units_per_row
        src_id = tl.load(SI + route, valid, 0).to(tl.int64)
        dst_id = tl.load(DI + route, valid, 0).to(tl.int64)
        source = tl.load(SOURCES + bank * TABLE_SIZE + src_id, valid, 0)
        col = (unit % units_per_row) * 16
        src_addr = (source + col).to(tl.pointer_type(tl.uint32))
        dst_addr = (destination + dst_id * units_per_row * 16 + col).to(tl.pointer_type(tl.uint32))
        value = tl.load(src_addr[:, None] + words[None, :], valid[:, None], 0)
        tl.store(dst_addr[:, None] + words[None, :], value, valid[:, None])

def gather(descriptor, dst_ids, src_ids, count, blocks=8):
    dst, sources, feats = descriptor
    _owner_gather[(len(dst), blocks)](
        dst, sources, feats, dst_ids, src_ids, count,
        sources.shape[1], blocks, 1024, num_warps=8)


@triton.jit(do_not_specialize=['BASE'])
def _ensure(IDS, MAP, DYNAMIC_IDS, USAGE, STEP, COPY_SRC, COPY_DST, COUNT,
            STATS, BASE, DYNAMIC_BEGIN: tl.constexpr,
            PROTECTED_BEGIN: tl.constexpr, DYNAMIC_SIZE: tl.constexpr,
            K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
            COLLECT_STATS: tl.constexpr):
    step = tl.load(STEP) + 1
    tl.store(STEP, step)
    k = tl.arange(0, BK)
    valid = k < K
    ids = tl.load(IDS + k, valid, 0).to(tl.int64)
    global_ids = ids + BASE
    slots = tl.load(MAP + global_ids, valid, -1)
    hits = valid & (slots >= 0)
    dynamic_hits = hits & (slots >= DYNAMIC_BEGIN) & (slots < PROTECTED_BEGIN)
    tl.store(USAGE + slots - DYNAMIC_BEGIN, step, dynamic_hits)
    same = (global_ids[:, None] == global_ids[None, :]) & (k[:, None] > k[None, :]) & valid[:, None] & valid[None, :]
    first = valid & (tl.sum(same.to(tl.int32), 1) == 0)
    first_miss = first & (slots < 0)
    miss = valid & (slots < 0)
    rank = tl.sum(((global_ids[None, :] < global_ids[:, None]) & first_miss[None, :]).to(tl.int32), 1)
    n = tl.sum(first_miss.to(tl.int32))
    tl.store(COUNT, n.to(tl.int64))
    out = slots
    if n > 0:
        tl.debug_barrier()
        c = tl.arange(0, BC)
        u = tl.load(USAGE + c, c < DYNAMIC_SIZE, 0x3FFFFFFFFFFFFFFF)
        u = tl.where((c >= DYNAMIC_SIZE) | (u == step), 0x3FFFFFFFFFFFFFFF, u)
        for i in range(n):
            victim = tl.argmin(u, 0).to(tl.int32)
            old = tl.load(DYNAMIC_IDS + victim)
            if old >= 0:
                tl.store(MAP + old, -1)
            incoming = tl.sum(tl.where(first_miss & (rank == i), global_ids, 0), 0)
            physical = victim + DYNAMIC_BEGIN
            tl.store(DYNAMIC_IDS + victim, incoming.to(tl.int32))
            tl.store(MAP + incoming, physical)
            tl.store(USAGE + victim, step)
            tl.store(COPY_SRC + i, (incoming - BASE).to(tl.int32))
            tl.store(COPY_DST + i, physical)
            out = tl.where(miss & (rank == i), physical, out)
            u = tl.where(c == victim, 0x3FFFFFFFFFFFFFFF, u)
    tl.store(IDS + k, out, valid)
    if COLLECT_STATS:
        s = tl.arange(0, 4)
        values = tl.where(s == 0, tl.sum(first.to(tl.int32), 0), tl.where(s == 1, n, 1))
        tl.atomic_add(STATS + s, values.to(tl.int64), s < 3)


def load_compact_banks(model_path, model_config, *, protected, dynamic, method, device, dtype, dummy=False,
                       parallel=None, decode_target='gpu', layer_residency=None, **kwargs):
    from freetoken.layers.quantization import QuantKind
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.expert_pieces import iter_expert_pieces

    if method.kind is not QuantKind.Q2_0 or dummy or decode_target != 'gpu' or layer_residency:
        raise ValueError('Protected residency requires real GPU Q2 expert banks')
    layers, experts = model_config.num_moe_layers, model_config.num_experts
    protected = tuple(protected)
    if len(set(protected)) != len(protected) or any(not (0 <= l < layers and 0 <= e < experts) for l, e in protected):
        raise ValueError('Invalid protected ownership set')
    # Decode can also borrow both full-layer prefill buffers. They already fit
    # the admitted route union even when no additional dynamic slots are used.
    if dynamic < 0:
        raise ValueError('Dynamic region cannot be negative')
    protected_begin = 2 * experts + dynamic
    slots = protected_begin + len(protected)
    layout = method.layout()
    if tuple(layout) != ('gate_up', 'down') or any(s.resident or s.dtype is not torch.uint8 for s in layout.values()):
        raise ValueError('Unexpected native Q2 bank layout')
    protected_slots = {pair: protected_begin + i for i, pair in enumerate(protected)}
    host_indices = [[-1] * experts for _ in range(layers)]
    host_counts = []
    for layer in range(layers):
        n = 0
        for expert in range(experts):
            if (layer, expert) not in protected_slots:
                host_indices[layer][expert] = n
                n += 1
        host_counts.append(n)
    # GPU owners are uploaded straight into the final pool, not a second staging bank.
    caches = {role: torch.empty((slots, *spec.shape), dtype=spec.dtype, device=device)
              for role, spec in layout.items()}
    sources = {role: [alloc_pinned_tensor(n, *spec.shape, dtype=spec.dtype) if n
                      else torch.empty((0, *spec.shape), dtype=spec.dtype)
                      for n in host_counts] for role, spec in layout.items()}
    written = torch.zeros((layers, experts), dtype=torch.bool)
    for layer, e0, e1, pieces in iter_expert_pieces(model_path, model_config, method.kind, parallel=False):
        if not (0 <= layer < layers and 0 <= e0 < e1 <= experts) or bool(written[layer, e0:e1].any()):
            raise ValueError('Missing/out-of-range/duplicated source piece')
        packed = {role: torch.empty((e1 - e0, *spec.shape), dtype=spec.dtype) for role, spec in layout.items()}
        if method.pack(pieces, packed):
            raise ValueError('Native Q2 banks should not return separate alpha tensors')
        host_rows = [e - e0 for e in range(e0, e1) if host_indices[layer][e] >= 0]
        gpu_rows = [e - e0 for e in range(e0, e1) if host_indices[layer][e] < 0]
        if host_rows:
            host_dst = torch.tensor([host_indices[layer][e0 + row] for row in host_rows], dtype=torch.int64)
            host_src = torch.tensor(host_rows, dtype=torch.int64)
        if gpu_rows:
            gpu_dst = torch.tensor([protected_slots[layer, e0 + row] for row in gpu_rows], device=device, dtype=torch.int64)
            gpu_src = torch.tensor(gpu_rows, dtype=torch.int64)
        for role in layout:
            if host_rows:
                sources[role][layer].index_copy_(0, host_dst, packed[role].index_select(0, host_src))
            if gpu_rows:
                caches[role].index_copy_(0, gpu_dst, packed[role].index_select(0, gpu_src).to(device))
        written[layer, e0:e1] = True
        if e1 == experts:
            logger.info_rank0(f'Protected Q2 layer {layer}: {host_counts[layer]} compact host rows')
    if not bool(written.all()):
        raise ValueError('Incomplete Q2 ownership load')
    torch.cuda.synchronize(device)
    initial_map = torch.full((layers, experts), -1, dtype=torch.int32, device='cpu')
    for (layer, expert), slot in protected_slots.items():
        initial_map[layer, expert] = slot
    ownership = Q2Ownership(sources=sources, caches=caches, host_indices=host_indices,
                   protected_slots=protected_slots, initial_map=initial_map,
                   slots=slots, dynamic=dynamic, protected_begin=protected_begin,
                   layers=layers, experts=experts, layout=layout,
                   host_bytes=sum(sum(t.numel() * t.element_size() for t in per_layer) for per_layer in sources.values()),
                   gpu_bank_bytes=sum(t.numel() * t.element_size() for t in caches.values()))
    return ExpertBanks('q2_0', sources, kind=method.kind, kernel=method.kernel.name, layout=layout), ownership


class ProtectedQ2Cache(OffloadMoeCache):
    def __init__(self, *args, ownership: Q2Ownership, decode_overlap=True, adapt_every=0,
                 adapt_decay=1.0, **kwargs):
        """单独持有decode copy流；prefill和decode不共享事件或借用缓冲协议。"""
        self.protected_payload = ownership
        if not 0 < adapt_decay <= 1:
            raise ValueError('Online Q2 heat decay must be in (0, 1]')
        self.adapt_every = adapt_every
        self.adapt_decay = adapt_decay
        super().__init__(*args, **kwargs)
        self.collect_decode_freq = bool(adapt_every)
        self.decode_copy_stream = torch.cuda.Stream(device=self.device) if decode_overlap else None

    def start_adaptation(self):
        """Drop capture warmup counts and bind current owner tables after all graphs are ready."""
        if not self.adapt_every:
            return None
        from .q2_adaptive import AdaptiveQ2Owners
        return AdaptiveQ2Owners(self, every=self.adapt_every, decay=self.adapt_decay)

    def set_bank_sources(self, sources, layer_residency=None):
        from freetoken.kernel.pinned import device_ptr
        payload = self.protected_payload
        if sources is not payload.sources or self.cache_size != payload.slots:
            raise ValueError('Protected cache must attach the exact loaded owner payload')
        if self.quant_format != 'q2_0' or self.decode_target != 'gpu' or self.cpu_layer_ids:
            raise ValueError('Protected cache is restricted to pure GPU Q2')
        self.bank_sources = sources
        self.bank_caches = payload.caches
        self.banks = [(sources[role], self.bank_caches[role]) for role in self.bank_schema]
        self.layer_residency = ['pinned'] * self.num_layers
        self.prefill_hit_d2d = False
        self._copy_fused_ok = True
        self._copy_dst_ptrs = torch.tensor([self.bank_caches[r].data_ptr() for r in self.bank_schema], device=self.device, dtype=torch.int64)
        row_bytes = [math.prod(self.bank_caches[r].shape[1:]) * self.bank_caches[r].element_size() for r in self.bank_schema]
        self._copy_feat_bytes = torch.tensor(row_bytes, device=self.device, dtype=torch.int64)
        self._owner_tables = []
        for layer in range(self.num_layers):
            pointers = []
            for bank, role in enumerate(self.bank_schema):
                host = sources[role][layer]
                host_base = device_ptr(host) if host.size(0) else 0
                gpu_base = self.bank_caches[role].data_ptr()
                pointers.append([host_base + payload.host_indices[layer][e] * row_bytes[bank]
                                 if payload.host_indices[layer][e] >= 0
                                 else gpu_base + payload.protected_slots[layer, e] * row_bytes[bank]
                                 for e in range(self.num_experts)])
            self._owner_tables.append(torch.tensor(pointers, device=self.device, dtype=torch.int64))
        self._protected_initial_map = payload.initial_map.to(self.device)
        reverse = torch.full((self.cache_size,), -1, dtype=torch.int32)
        for (layer, expert), slot in payload.protected_slots.items():
            reverse[slot] = layer * self.num_experts + expert
        self._protected_reverse_map = reverse.to(self.device)
        self._whole_ids = torch.arange(self.num_experts, device=self.device, dtype=torch.int32)
        self._whole_count = torch.full((1,), self.num_experts, device=self.device, dtype=torch.int64)
        self._whole_dst = [self._whole_ids + b * self.num_experts for b in (0, 1)]
        self.reset()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _gather(self, layer, dst, src, count, blocks):
        gather((self._copy_dst_ptrs, self._owner_tables[layer], self._copy_feat_bytes), dst, src, count, blocks)

    def ensure_experts(self, layer_id, expert_ids):
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        n = expert_ids.numel()
        p = self.protected_payload
        # The existing sparse-prefill path admits 16 tokens x 10 expert routes.
        if not (0 <= layer_id < self.num_layers) or not (0 < n <= 256):
            raise ValueError('Protected admission supports at most 256 routes per layer')
        if self.collect_decode_freq:
            self.decode_freq[layer_id].scatter_add_(0, expert_ids.view(-1).to(torch.int64),
                torch.ones(n, dtype=torch.int64, device=self.device))
        # decode/稀疏残余prefill可借用两个空闲prefill缓冲，永久所有者不变。
        begin, end = 0, 2 * self.num_experts if getattr(self, '_workspace_borrow_active', False) else p.protected_begin
        _ensure[(1,)](expert_ids, self.slot_for_id.view(-1), self.id_of_slot[begin:end],
                      self.usage[begin:end], self.step, self.src_indices, self.evict_slots,
                      self.num_indices, self.lru_stats[layer_id], layer_id * self.num_experts,
                      begin, end, end - begin, n, triton.next_power_of_2(n),
                      triton.next_power_of_2(end - begin), self.collect_stats, num_warps=4)

    def copy_missing(self):
        layer = self._pending_src_layer
        if layer is None:
            raise RuntimeError('No pending Q2 source layer')
        if self._pending_whole_layer:
            self._invalidate_prefill_buffer(0)
            self._gather(layer, self._whole_dst[0], self._whole_ids, self._whole_count, 64)
        else:
            self._gather(layer, self.evict_slots, self.src_indices, self.num_indices, 32)

    def materialize_layer(self, layer_id):
        self._pending_src_layer = layer_id
        self._pending_whole_layer = True

    def _invalidate_prefill_buffer(self, buffer_id):
        # 覆盖prefill缓冲前撤销借用副本映射；只清理这两个可丢弃区域。
        super()._invalidate_prefill_buffer(buffer_id)

    def prefetch_prefill_layer(self, layer_id):
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError('Invalid source layer')
        buffer = layer_id % 2
        if self._prefill_buffer_layer[buffer] == layer_id:
            return
        if self._prefill_buffer_layer[buffer] is not None and not self._prefill_buffer_released[buffer]:
            raise RuntimeError('Prefill buffer reused before compute released it')
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer])
            self._invalidate_prefill_buffer(buffer)
            self._gather(layer_id, self._whole_dst[buffer], self._whole_ids, self._whole_count, 64)
            self.prefill_ready_events[buffer].record(self.prefill_copy_stream)
        self._prefill_buffer_layer[buffer] = layer_id
        self._prefill_buffer_released[buffer] = False

    def reset(self):
        if self.decode_copy_stream is not None:
            self.decode_copy_stream.synchronize()
        if self.prefill_copy_stream is not None:
            self.prefill_copy_stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self.slot_for_id.copy_(self._protected_initial_map)
        self.id_of_slot.copy_(self._protected_reverse_map)
        self.usage.zero_()
        self.step.zero_()
        self.expert_recency.fill_(-1)
        self.num_indices.zero_()
        self._pending_src_layer = None
        self._pending_whole_layer = False
        # All users of the borrowed prefill views finished before these tags reset.
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]

    def validate_rebuild(self, cache_size):
        """Reject an owner-pool resize before the engine destroys live graphs."""
        super().validate_rebuild(cache_size)
        if cache_size != self.protected_payload.slots:
            raise ValueError('Protected Q2 expert capacity is fixed; reload the model to change it')

    def rebuild(self, cache_size):
        """Same-capacity rebuild resets dynamic copies and retains all owner addresses."""
        self.validate_rebuild(cache_size)
        self.reset()
        self.reset_stats()
        self.decode_freq.zero_()
        self.prefill_hit_rows = self.prefill_total_rows = 0
