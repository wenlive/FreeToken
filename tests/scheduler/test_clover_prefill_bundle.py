"""A first denoised block must be wholly on the host before prefix insertion."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.scheduler import Scheduler


def test_prefill_bundle_publishes_one_complete_prefix():
    req = Req(
        torch.tensor([7, 8, 9], dtype=torch.int32),
        0,
        0,
        20,
        1,
        SamplingParams(max_tokens=20),
        None,
    )
    req.cached_len, req.device_len = 5, 6
    batch = Batch([req], "prefill")
    batch.mtp_token_ids = torch.tensor([10, 11, 12], dtype=torch.int32)
    snapshots = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = SimpleNamespace(
        lazy_free_region=nullcontext,
        cache_req=lambda r, finished: snapshots.append(
            (r.input_ids.tolist(), r.cached_len)
        ),
    )
    scheduler.finished_reqs = set()
    scheduler.eos_token_ids = set()
    scheduler.toolcall_anchor_id = None
    scheduler.decode_manager = SimpleNamespace(running_reqs={req})
    scheduler.prefill_manager = SimpleNamespace(pending_list=[])
    scheduler.config = SimpleNamespace(page_size=1)
    scheduler._kv_usage_pages = lambda: (0, 20)
    scheduler._mamba_slot_usage = lambda: None
    scheduler._swa_token_usage = lambda: None
    scheduler._gpu_mem_bytes = lambda: 0
    scheduler.status_reporter = SimpleNamespace(
        report_batch=lambda *args, **kwargs: None
    )
    sent = []
    scheduler.send_result = sent.extend
    output = (None, None, SimpleNamespace(synchronize=lambda: None))
    Scheduler._process_last_data(scheduler, (SimpleNamespace(batch=batch), output))
    assert snapshots == [([7, 8, 9, 10, 11, 12], 5)]
    assert [reply.next_token for reply in sent] == [10, 11, 12]
