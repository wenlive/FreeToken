"""Abort vs overlap-scheduling races.

Under overlap scheduling a batch launches one iteration before _process_last_data drains
it, and an abort message is processed in between. Freeing the request's resources inside
the abort handler while its forward is in flight used to corrupt state: the hybrid
prefill-commit dereferenced the None'd GDN ping-pong slots (TypeError killed the
scheduler); plain radix would silently re-read the freed page-table row.

The scheduler now uses the SGLang-style single-owner design: the abort handler frees
immediately ONLY when the request has no forward in flight (not in ``self._last_data``'s
batch); otherwise it just sets ``req.aborted`` and _process_last_data frees the request
when the batch drains, after copy_done.synchronize(). A ``table_idx != -1`` sentinel on
the prefix-commit remains as defense-in-depth.

Tests drive the real (unbound) Scheduler methods against CPU-built hybrid managers.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.message import AbortBackendMsg
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

UID = 2


def test_context_clipping_does_not_shrink_another_request_with_shared_sampling():
    """Offline batches can share one SamplingParams; each prompt owns its clipped budget."""
    from freetoken.message import UserMsg

    _, _, _, _, pm, _, scheduler = _setup()
    scheduler.engine = SimpleNamespace(max_seq_len=64, encoder_cache=None)
    shared = SamplingParams(max_tokens=32)
    long = UserMsg(uid=10, input_ids=torch.arange(56, dtype=torch.int32), sampling_params=shared)
    short = UserMsg(uid=11, input_ids=torch.arange(8, dtype=torch.int32), sampling_params=shared)
    Scheduler._process_one_msg(scheduler, long)
    Scheduler._process_one_msg(scheduler, short)
    pending = {req.uid: req for req in pm.pending_list}
    assert pending[10].output_len == 8
    assert pending[11].output_len == 32
    assert shared.max_tokens == 32


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _setup():
    """Hybrid managers + a stub Scheduler `self` for the real unbound methods."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    sent = []
    stub = SimpleNamespace(
        cache_manager=cm,
        table_manager=tm,
        decode_manager=dm,
        prefill_manager=pm,
        finished_reqs=set(),
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        send_result=sent.extend,
        _kv_usage_pages=cm.page_usage,
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
        _pending_abort_acks=set(),
        _last_data=None,
    )
    stub._free_req_resources = lambda req: Scheduler._free_req_resources(stub, req)
    return pool, cm, tm, dm, pm, sent, stub


def _launch_req(pool, cm, tm, prompt, *, cls=Req, track_seqlen=None):
    """A launched (forward in flight) hybrid req: handle locked, pages allocated,
    GDN slots held, cached_len advanced -- the state _process_last_data will drain."""
    mr = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=len(prompt),
                                      ))
    req = cls(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=4,
              uid=UID, sampling_params=SamplingParams(max_tokens=4),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    cm.lock(mr.cuda_handle)
    cm.allocate_paged([req])
    req.complete_one()
    req.mamba_last_track_seqlen = track_seqlen
    return req


def _as_last_data(batch):
    return (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42], dtype=torch.int32),
         SimpleNamespace(synchronize=lambda: None)),
    )


def test_abort_final_mtp_bundle_owns_request_after_decode_filter():
    """The final verified window still owns pages until its terminal drain."""
    from freetoken.message import DetokenizeMsg, ErrorReplyMsg

    pool, cm, tm, dm, pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.tensor([11, 12, 13, 14], dtype=torch.int32))
    req.append_host(torch.tensor([20], dtype=torch.int32))
    cm.allocate_paged([req], reserve_tokens=2)
    req.cached_len, req.device_len = 7, 8
    batch = Batch([req], 'decode')
    batch.mtp_token_ids = torch.tensor([21, 22, 23], dtype=torch.int32)
    dm.filter_reqs([req])
    assert not dm.runnable
    last = _as_last_data(batch)
    stub._last_data = last
    free_before = len(cm.free_slots)

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert req.aborted
    assert req.table_idx >= 0 and len(cm.free_slots) == free_before
    Scheduler._process_last_data(stub, last)
    Scheduler._flush_abort_acks(stub)
    assert not any(isinstance(msg, DetokenizeMsg) for msg in sent)
    assert len(sent) == 1 and isinstance(sent[0], ErrorReplyMsg)
    assert sent[0].uid == UID and sent[0].error == 'request aborted'
    assert req.input_ids.tolist() == [11, 12, 13, 14, 20, 21, 22, 23]
    assert req.table_idx == -1 and tm.available_size == 4
    cm.check_integrity()


def test_abort_inflight_final_chunk_marks_then_drains():
    """Abort while the final prefill chunk (plain Req, already in running_reqs) is in
    flight: the handler only marks; the same iteration's drain frees exactly once."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    batch = Batch(reqs=[req], phase="prefill")
    dm.filter_reqs(batch.reqs)                  # _forward: joins running_reqs at launch
    stub._last_data = _as_last_data(batch)      # overlap_loop exposes the un-drained batch

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert req.aborted and req.table_idx != -1  # marked, NOT freed under the forward
    assert req.mamba_ping_pong is not None
    assert req not in dm.running_reqs
    assert UID in stub._pending_abort_acks
    free_after_mark = pool.num_free_slots

    Scheduler._process_last_data(stub, stub._last_data)
    assert req.table_idx == -1                  # freed at the drain point
    assert pool.num_free_slots > free_after_mark
    assert req in stub.finished_reqs
    assert sent == []                           # no DetokenizeMsg: abort ack stays terminal
    cm.check_integrity()


def test_abort_inflight_intermediate_chunk_marks_then_drains():
    """Abort mid-chunked-prefill: prefill_manager.abort_req pops the pending continuation
    (no next chunk launches) and returns the in-flight ChunkedReq, which is marked and
    freed when its batch drains."""
    pool, cm, tm, _dm, pm, sent, stub = _setup()
    prompt = torch.arange(1, 13, dtype=torch.int32)
    chunk = _launch_req(pool, cm, tm, prompt[:8], cls=ChunkedReq)
    pending = PendingReq(uid=UID, input_ids=prompt,
                         sampling_params=SamplingParams(max_tokens=4))
    pending.chunked_req = chunk
    pm.pending_list = [pending]
    batch = Batch(reqs=[chunk], phase="prefill")
    stub._last_data = _as_last_data(batch)

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert pm.pending_list == []                # continuation gone: no next chunk
    assert chunk.aborted and chunk.table_idx != -1

    Scheduler._process_last_data(stub, stub._last_data)
    assert chunk.table_idx == -1
    assert sent == []                           # chunks never reply
    cm.check_integrity()


def test_abort_starved_decode_req_frees_immediately():
    """A request with no forward in flight (e.g. a decode req starved behind a long
    chunked prefill) is freed by the abort handler right away -- deferring would leak
    until its next batch, which strict prefill-priority puts arbitrarily far away."""
    pool, cm, tm, dm, _pm, _sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32))
    dm.filter_reqs([req])
    # the un-drained batch belongs to some other request's prefill
    stub._last_data = (SimpleNamespace(batch=SimpleNamespace(reqs=[])), None)
    base_free = pool.num_free_slots

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert not req.aborted
    assert req.table_idx == -1                  # freed immediately, no drain needed
    assert pool.num_free_slots > base_free
    assert req not in dm.running_reqs
    cm.check_integrity()


def test_prefix_commit_sentinel_guard():
    """Defense-in-depth: even if some future path frees a req early (bypassing the
    aborted mark), the finished=False prefix-commit must skip a freed req instead of
    dereferencing its None'd GDN slots (the original crash, cache.py _cache_req_hybrid)."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    batch = Batch(reqs=[req], phase="prefill")
    dm.filter_reqs(batch.reqs)

    aborted = dm.abort_req(UID)
    assert aborted is req
    Scheduler._free_req_resources(stub, aborted)   # freed WITHOUT the aborted mark
    assert req.table_idx == -1 and req.mamba_ping_pong is None
    free_after_abort = pool.num_free_slots

    Scheduler._process_last_data(stub, _as_last_data(batch))  # pre-guard: TypeError

    assert pool.num_free_slots == free_after_abort  # nothing double-freed
    cm.check_integrity()
    assert [m.uid for m in sent] == [UID]  # un-marked path still publishes the token


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


def test_post_terminal_overlap_step_is_dropped():
    """Overlap scheduling launches one more decode step for a request that already
    terminated (filter_reqs keeps it while output budget remains). The extra drain
    must not append its token, emit a second DetokenizeMsg, or free twice."""
    from freetoken.message import DetokenizeMsg

    pool, cm, tm, dm, _pm, sent, stub = _setup()
    stub.eos_token_ids = {42}  # the drained token (42) finishes the request by EOS
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    dm.filter_reqs([req])

    Scheduler._process_last_data(stub, _as_last_data(Batch(reqs=[req], phase="prefill")))
    assert req in stub.finished_reqs and req.table_idx == -1
    terminal = [m for m in sent if isinstance(m, DetokenizeMsg)]
    assert len(terminal) == 1 and terminal[0].finished

    # The overlap extra step: the same req sits in the next batch's drain.
    output_len_before = req.output_len
    Scheduler._process_last_data(stub, _as_last_data(Batch(reqs=[req], phase="decode")))
    assert [m for m in sent if isinstance(m, DetokenizeMsg)] == terminal  # no 2nd msg
    assert req.output_len == output_len_before                           # no append
    cm.check_integrity()


@pytest.mark.parametrize('history,bundle,stop,ignore_eos,count,reason', [
    ([0], [1, 2, 2], ['stop'], False, 1, 'stop'),
    ([2], [4, 2, 2], [], False, 1, 'stop'),
    ([2], [4, 2, 2], [], True, 3, 'length'),
    ([2, 2], [2, 2, 2], [], False, 2, 'length'),
])
def test_verified_bundle_uses_native_stop_eos_and_length_boundaries(
    history, bundle, stop, ignore_eos, count, reason,
):
    """Drive the real CPU cache lifecycle and tokenizer through clipping and scheduler delivery."""
    from tokenizers import Tokenizer, decoders, models
    from freetoken.message import DetokenizeMsg

    pool, cm, tm, dm, _, sent, scheduler = _setup()
    tokenizer = Tokenizer(models.WordLevel({'st': 0, 'op': 1, 'x': 2, 'prompt': 3, '<eos>': 4}))
    tokenizer.decoder = decoders.Fuse()
    scheduler.tokenizer = tokenizer
    scheduler.eos_token_ids = {4}
    scheduler._match_stop_str = lambda req: Scheduler._match_stop_str(scheduler, req)
    # The prompt itself contains a stop string; it must not terminate generated text.
    req = _launch_req(pool, cm, tm, torch.tensor([0, 1, 3], dtype=torch.int32))
    req.sampling_params.stop_strs = stop
    req.sampling_params.ignore_eos = ignore_eos
    req.append_host(torch.tensor(history[:1], dtype=torch.int32))
    for token in history[1:]:
        cm.allocate_paged([req])
        req.complete_one()
        req.append_host(torch.tensor([token], dtype=torch.int32))
    before = req.input_ids.clone()
    limit = Scheduler._mtp_delivery_limit(scheduler, req, bundle)
    assert limit.count == count and limit.finished
    assert torch.equal(req.input_ids, before)
    batch = Batch([req], 'decode')
    batch.mtp_token_ids = torch.tensor(bundle[:limit.count], dtype=torch.int32)
    cm.allocate_paged([req], reserve_tokens=count)
    # The runtime has committed the accepted rows before the scheduler drains them.
    req.cached_len = len(req.input_ids) + count - 1
    req.device_len = len(req.input_ids) + count
    dm.filter_reqs([req])
    Scheduler._process_last_data(scheduler, _as_last_data(batch))
    replies = [message for message in sent if isinstance(message, DetokenizeMsg)]
    assert [message.next_token for message in replies] == bundle[:count]
    assert replies[-1].finished and replies[-1].finish_reason == reason
    assert req.table_idx == -1 and req.mamba_ping_pong is None
    cm.check_integrity()
