"""Lend disposable Q2 payload rows to eager prefill without changing owner bytes."""
from collections import deque
import ctypes

import torch

from freetoken.kernel.pinned import _load_pinned_extension
from freetoken.moe.offload_cache import OffloadMoeCache


class PrefillWorkspace:
    """Keep bank ownership and allocator lifetime together for one engine."""

    def __init__(self, cache, device):
        if not hasattr(cache, 'protected_payload') or cache.protected_payload.dynamic != 1024:
            raise ValueError('Prefill workspace requires protected Q2 with 1024 dynamic rows')
        self.cache, self.device = cache, device
        self.failed = False
        self.records = deque(maxlen=256)
        extension = _load_pinned_extension()
        self.native = ctypes.CDLL(extension.__file__)
        self.native.freetoken_workspace_setup.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.native.freetoken_workspace_setup.restype = ctypes.c_int
        self.native.freetoken_workspace_stat.argtypes = [ctypes.c_int]
        self.native.freetoken_workspace_stat.restype = ctypes.c_uint64
        self.allocator = torch.cuda.memory.CUDAPluggableAllocator(
            extension.__file__, 'freetoken_workspace_alloc', 'freetoken_workspace_free')

    def check_usable(self):
        """Reject all later native/MTP forwards after a failed loan; unload this engine."""
        if self.failed:
            raise RuntimeError('Prefill workspace failed; unload the engine before another request')

    def statistics(self):
        """Read bounded allocator counters only at the prefill transaction boundary."""
        return [self.native.freetoken_workspace_stat(i) for i in range(9)]

    @staticmethod
    def _preserve_metadata(batch, regions):
        """Keep per-forward QSA plans alive on the ordinary allocator after the loan."""
        ranges = [(r.data_ptr(), r.data_ptr() + r.numel()) for r in regions]
        names = []
        for name, value in tuple(vars(batch.attn_metadata).items()):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                base = value.untyped_storage().data_ptr()
                if any(lo <= base < hi for lo, hi in ranges):
                    setattr(batch.attn_metadata, name, value.clone())
                    names.append(name)
        return names

    def forward(self, function, batch):
        """Drain readers, lend only dynamic copies, and release every loan before decode."""
        self.check_usable()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prefill workspace borrowing is eager-only')
        cache = self.cache
        begin, end = 2 * cache.num_experts, cache.protected_payload.protected_begin
        if end - begin != 1024:
            raise ValueError('Prefill workspace geometry changed')
        for stream in (cache.decode_copy_stream, cache.prefill_copy_stream):
            if stream is not None:
                stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        for buffer in (2, 3):
            OffloadMoeCache._invalidate_prefill_buffer(cache, buffer)
        banks = [cache.protected_payload.caches[role] for role in cache.bank_schema]
        if len(banks) != 2:
            raise ValueError('Prefill workspace requires the gate/up and down banks')
        regions = [bank[begin:end].view(-1) for bank in banks]
        status = self.native.freetoken_workspace_setup(
            regions[0].data_ptr(), regions[0].numel(), regions[1].data_ptr(), regions[1].numel())
        if status:
            raise RuntimeError(f'Previous prefill workspace remains live: {status}')
        pool = torch.cuda.MemPool(allocator=self.allocator.allocator())
        cache._workspace_borrow_active = True
        try:
            with torch.cuda.use_mem_pool(pool):
                borrowed_logits = function()
            logits = borrowed_logits.clone()
            del borrowed_logits
            preserved = self._preserve_metadata(batch, regions)
            torch.cuda.synchronize(self.device)
        except BaseException:
            # A traceback can retain loan tensors. Keep their backing banks and pool alive,
            # and block future native/MTP execution instead of reusing those bytes.
            self.failed = True
            self._failed_regions, self._failed_pool = regions, pool
            raise
        finally:
            cache._workspace_borrow_active = False
        before = self.statistics()
        del pool
        torch.cuda.empty_cache()
        after = self.statistics()
        self.records.append(dict(tokens=len(batch.input_ids), loan_bytes=sum(r.numel() for r in regions),
            before_release=before, after_release=after, preserved_metadata=preserved,
            allocated=torch.cuda.memory_allocated(self.device), reserved=torch.cuda.memory_reserved(self.device)))
        if after[8] or after[7] or bool((cache.id_of_slot[begin:end] >= 0).any()):
            self.failed = True
            self._failed_regions = regions
            raise RuntimeError(f'Prefill workspace did not return every disposable row: {self.records[-1]}')
        return logits
