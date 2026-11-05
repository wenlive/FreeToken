"""A first denoised block must be wholly on the host before prefix insertion."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.scheduler import Scheduler


def request_runtime(prompt=(3, 4, 5)):
    """CPU cache ownership with a deterministic stand-in for the model forward."""
    from collections import deque
    import weakref
    from freetoken.models.clover_moe.runtime import CloverRuntime
    from tests.scheduler.test_abort_inflight_prefill import _setup, _launch_req

    pool, manager, tables, decode, prefill, replies, scheduler = _setup()
    req = _launch_req(pool, manager, tables, torch.tensor(prompt, dtype=torch.int32))
    req.device_len = len(req.input_ids)
    req.output_len = 16
    req.max_device_len = req.prompt_len + req.output_len
    req._alloc_ids_buf()
    req.sampling_params = SamplingParams(clover_mode="denoise", clover_width=8, max_tokens=16)
    req.device_len = len(req.input_ids)
    req.cached_len = req.device_len - 1
    runtime = CloverRuntime.__new__(CloverRuntime)
    runtime.states = weakref.WeakKeyDictionary()
    runtime._attempts = weakref.WeakKeyDictionary()
    runtime._cursor_warnings = weakref.WeakSet()
    runtime.rounds = deque()
    runtime.transactions = None
    token_pool = torch.zeros_like(manager.page_table)
    token_pool[req.table_idx, :len(prompt)] = req.input_ids
    from freetoken.engine.engine import ForwardOutput

    def native(batch, args):
        for r in batch.reqs:
            r.complete_one()
        token = (batch.input_ids[-1:] + 1).to(torch.int32)
        return ForwardOutput(token, token, SimpleNamespace(synchronize=lambda: None))

    runtime.engine = SimpleNamespace(device="cpu", page_table=manager.page_table,
        linear_state_pool=pool, sampler=SimpleNamespace(prepare=lambda batch: None),
        forward_batch=native, stream=SimpleNamespace(synchronize=lambda: None, wait_stream=lambda stream: None),
        config=SimpleNamespace(clover_mode="denoise", clover_width=8,
            clover_threshold=.95, clover_max_width=32, max_running_req=1),
        moe_offload_cache=None)
    scheduler.engine = runtime.engine
    scheduler.engine.mtp_runtime = runtime
    scheduler.token_pool = token_pool
    scheduler._mtp_delivery_limit = lambda r, values: SimpleNamespace(count=len(values), finished=False)
    decode.running_reqs.add(req)
    batch = Batch([req], "decode")
    batch.input_ids = torch.tensor([prompt[-1]], dtype=torch.int32)
    batch.padded_reqs = [req]
    return runtime, scheduler, manager, tables, req, batch, replies


def test_denoise_capacity_fallback_precedes_preview_and_preserves_clean_boundary():
    runtime, scheduler, manager, tables, req, batch, replies = request_runtime()
    # Reserve the other pages as an independent live owner, rather than faking
    # the manager's available_size bookkeeping.
    held = manager._allocate(manager.num_pages - 3)
    runtime.block_graphs = SimpleNamespace(clean_owner=req, clean_position=req.device_len)
    runtime._blocks = lambda: pytest.fail("capacity fallback entered preview")
    before = manager.available_size
    old_device = req.device_len
    req.output_len = 1
    req.max_device_len = old_device + 1
    result = runtime.forward(batch, manager, scheduler.token_pool, set(), None, None)
    assert result.next_tokens_cpu.tolist() == [6]
    assert scheduler.token_pool[req.table_idx, old_device] == 6
    assert runtime.rounds[-1]["fallback"] == "kv_capacity"
    assert runtime.block_graphs.clean_position != req.device_len
    assert manager.available_size == before
    manager._free(held)
    Scheduler._process_last_data(scheduler, (SimpleNamespace(batch=batch), result))
    assert req.table_idx == -1 and replies[-1].finish_reason == "length"
    manager.check_integrity()
    assert manager.available_size == manager.num_pages


@pytest.mark.parametrize("where", ["eligible", "forward", "native"])
def test_request_error_frees_owned_pages_and_next_request_matches_alone(where):
    from freetoken.engine.speculative import SpeculativeRequestError
    from freetoken.scheduler.scheduler import ForwardInput

    runtime, scheduler, manager, tables, req, batch, replies = request_runtime()
    base_pages = manager.num_pages
    base_slots = manager.linear_state_pool.num_slots - 1

    def fail(*args):
        manager.allocate_paged([req], reserve_tokens=2)
        raise SpeculativeRequestError("injected request failure", code="invalid_state")

    runtime.eligible = fail if where == "eligible" else lambda batch: where != "native"
    runtime._forward = fail
    if where == "native":
        runtime.engine.forward_batch = fail
    indices = (torch.tensor([req.table_idx]), torch.tensor([req.device_len - 1]))
    forward = ForwardInput(batch, None, indices, indices)
    assert Scheduler._forward(scheduler, forward) is None
    assert req.table_idx == -1
    assert req not in scheduler.decode_manager.running_reqs
    assert replies[-1].code == "invalid_state"
    assert manager.available_size == base_pages
    assert manager.linear_state_pool.num_free_slots == base_slots
    manager.check_integrity()
    # The next scheduler invocation executes the same deterministic native
    # tokens as a clean scheduler, using newly allocated real cache resources.
    reference, reference_scheduler, _, _, reference_req, reference_batch, _ = request_runtime((8, 9, 10))
    reference.eligible = lambda batch: False
    from tests.scheduler.test_abort_inflight_prefill import _launch_req
    next_req = _launch_req(manager.linear_state_pool, manager, tables,
                           torch.tensor([8, 9, 10], dtype=torch.int32))
    next_req.device_len = len(next_req.input_ids)
    next_req.cached_len = next_req.device_len - 1
    next_batch = Batch([next_req], "decode")
    scheduler.token_pool[next_req.table_idx, :3] = next_req.input_ids
    runtime.eligible = lambda batch: False
    runtime.engine.forward_batch = reference.engine.forward_batch
    actual_ids, reference_ids = [], []
    for _ in range(3):
        for target, current, current_batch, values in [
            (scheduler, next_req, next_batch, actual_ids),
            (reference_scheduler, reference_req, reference_batch, reference_ids),
        ]:
            read = (torch.tensor([current.table_idx]), torch.tensor([current.device_len - 1]))
            write = (torch.tensor([current.table_idx]), torch.tensor([current.device_len]))
            output = Scheduler._forward(target, ForwardInput(current_batch, None, read, write))
            values.extend(output.next_tokens_cpu.tolist())
    assert actual_ids == reference_ids == [11, 12, 13]
    runtime.abort_request(next_req)
    scheduler._free_req_resources(next_req)
    manager.check_integrity()
    assert manager.available_size == base_pages


def test_ordinary_runtime_error_is_not_swallowed():
    from freetoken.scheduler.scheduler import ForwardInput

    runtime, scheduler, manager, tables, req, batch, replies = request_runtime()
    runtime.eligible = lambda batch: True

    def fail(*args):
        raise RuntimeError("unrecoverable model failure")

    runtime._forward = fail
    indices = (torch.tensor([req.table_idx]), torch.tensor([req.device_len - 1]))
    with pytest.raises(RuntimeError, match="unrecoverable model failure"):
        Scheduler._forward(scheduler, ForwardInput(batch, None, indices, indices))
    assert not replies


def test_request_rollback_keeps_an_existing_shared_prefix_and_returns_only_owned_pages():
    from freetoken.models.clover_moe.graph import CloverBlockGraphs

    runtime, scheduler, manager, tables, previous, batch, replies = request_runtime()
    pool = manager.linear_state_pool
    pool.conv_states[:, previous.linear_slot_idx].fill_(11)
    pool.recurrent_states[:, previous.linear_slot_idx].fill_(13)
    scheduler._free_req_resources(previous)
    prompt = torch.tensor([3, 4, 9], dtype=torch.int32)
    match = manager.match_req(SimpleNamespace(input_ids=prompt, input_len=3))
    assert match.cuda_handle.cached_len == 2
    req = Req(prompt, tables.allocate(), 2, 16, 99, SamplingParams(), match.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    pool.copy_from(match.mamba_value, req.linear_slot_idx)
    manager.page_table[req.table_idx, :2].copy_(match.cuda_handle.get_matched_indices())
    manager.lock(match.cuda_handle)
    manager.allocate_paged([req])
    graphs = CloverBlockGraphs.__new__(CloverBlockGraphs)
    graphs.engine, graphs.clean = runtime.engine, None
    graphs.clean_owner = graphs.clean_position = None
    graphs.prepare_clean(req)
    runtime.block_graphs = graphs
    runtime._attempts[req] = SimpleNamespace(manager=manager, reserved_len=req.mtp_reserved_len,
                                           device_len=req.device_len, clean_ready=True)
    manager.allocate_paged([req], reserve_tokens=2)
    pool.conv_states[:, req.linear_slot_idx].fill_(99)
    pool.recurrent_states[:, req.linear_slot_idx].fill_(99)
    runtime.abort_request(req)
    scheduler._free_req_resources(req)
    again = manager.match_req(SimpleNamespace(input_ids=prompt, input_len=3))
    assert again.cuda_handle.cached_len == 2 and again.mamba_value == match.mamba_value
    assert torch.all(pool.conv_states[:, again.mamba_value] == 11)
    assert torch.all(pool.recurrent_states[:, again.mamba_value] == 13)
    manager.check_integrity()
    assert manager.available_size == manager.num_pages


@pytest.mark.parametrize("mode", ["selfspec", "denoise"])
def test_empty_delivery_is_a_request_error_and_releases_tentative_pages(monkeypatch, mode):
    from freetoken.models.clover_moe.generation import PENALTY_DEFAULTS
    from freetoken.models.clover_moe.graph import CloverBlockGraphs
    from freetoken.models.clover_moe import generation
    from freetoken.kernel.triton import clover_pick, clover_verify
    from freetoken.scheduler.scheduler import ForwardInput

    runtime, scheduler, manager, tables, req, batch, replies = request_runtime()
    req.sampling_params.clover_mode = mode
    runtime.engine.config.model_config = SimpleNamespace(vocab_size=17)
    runtime.engine.config.clover_accept_rule = "exact"
    runtime.engine.config.clover_accept_ratio = 1.0
    runtime.engine.config.clover_accept_delta = 0.0
    runtime.generation = SimpleNamespace(**PENALTY_DEFAULTS)
    runtime.mask_id, runtime.stop_ids = 16, [15]
    graphs = CloverBlockGraphs.__new__(CloverBlockGraphs)
    graphs.engine, graphs.clean = runtime.engine, None
    graphs.clean_owner = graphs.clean_position = None
    runtime.block_graphs = graphs
    runtime.eligible = lambda batch: True
    runtime._advance_without_sampling = lambda batch: None
    runtime._preview = lambda req, tokens: torch.zeros(len(tokens), 17)

    def verify(req, tokens, capacity):
        runtime.transactions = SimpleNamespace(width=capacity, valid_width=len(tokens), written={0})
        return torch.zeros(capacity, 17)

    runtime._verify = verify
    monkeypatch.setattr(clover_pick, "pick_with_confidence",
                        lambda scores, *args: (torch.zeros(len(scores), dtype=torch.int32), None))
    monkeypatch.setattr(clover_verify, "verify_and_deliver",
                        lambda *args, **kwargs: torch.tensor([0, 7]))
    monkeypatch.setattr(generation, "resolve_low_confidence",
                        lambda *args, **kwargs: (torch.full((8,), 7, dtype=torch.int32), [8]))
    # runtime.py imports this resolver by value.
    from freetoken.models.clover_moe import runtime as module
    monkeypatch.setattr(module, "resolve_low_confidence", generation.resolve_low_confidence)
    scheduler._mtp_delivery_limit = lambda req, values: SimpleNamespace(count=0, finished=True)
    indices = (torch.tensor([req.table_idx]), torch.tensor([req.device_len - 1]))
    assert Scheduler._forward(scheduler, ForwardInput(batch, None, indices, indices)) is None
    assert replies[-1].code == "invalid_state"
    assert "delivery is empty" in replies[-1].error
    assert req.table_idx == -1 and manager.available_size == manager.num_pages
    assert graphs.clean_owner is None
    if runtime.transactions is not None:
        assert runtime.transactions.width == 0 and not runtime.transactions.written
    manager.check_integrity()


@pytest.mark.parametrize("overlap", [False, True])
def test_request_error_does_not_drain_a_missing_forward_or_stop_the_loop(overlap):
    from freetoken.engine.speculative import SpeculativeRequestError
    from freetoken.scheduler.scheduler import ForwardInput
    from contextlib import nullcontext

    runtime, scheduler, manager, tables, req, batch, replies = request_runtime()

    def fail(batch):
        raise SpeculativeRequestError("KV capacity changed during request", code="kv_capacity_exhausted")

    runtime.eligible = fail
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.stream = SimpleNamespace(wait_stream=lambda stream: None)
    scheduler.receive_msg = lambda blocking: []
    scheduler._pending_rebuild = None
    indices = (torch.tensor([req.table_idx]), torch.tensor([req.device_len - 1]))
    scheduled = [ForwardInput(batch, None, indices, indices), None]
    scheduler._schedule_next_batch = lambda: scheduled.pop(0)
    scheduler._restore_linear_states = lambda batch: None
    scheduler._forward = lambda forward: Scheduler._forward(scheduler, forward)
    drained = []
    scheduler._process_last_data = drained.append
    scheduler._flush_abort_acks = lambda: None
    if overlap:
        assert Scheduler.overlap_loop(scheduler, None) is None
        assert Scheduler.overlap_loop(scheduler, None) is None
    else:
        Scheduler.normal_loop(scheduler)
        Scheduler.normal_loop(scheduler)
    assert drained == [None, None]
    assert len(replies) == 1 and replies[0].code == "kv_capacity_exhausted"
    manager.check_integrity()


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


def admission_scheduler(**config_overrides):
    from freetoken.models.clover_moe.runtime import CloverRuntime

    config = SimpleNamespace(clover_mode="causal", clover_width=8,
        clover_threshold=.95, clover_max_width=32, max_running_req=1)
    config.__dict__.update(config_overrides)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(config=config, max_seq_len=64,
        moe_offload_cache=None, encoder_cache=None)
    runtime = CloverRuntime.__new__(CloverRuntime)
    runtime.engine = scheduler.engine
    scheduler.engine.mtp_runtime = runtime
    scheduler.cache_manager = SimpleNamespace(is_hybrid=True)
    admitted, replies = [], []
    scheduler.prefill_manager = SimpleNamespace(add_one_req=admitted.append)
    scheduler.send_result = replies.extend
    return scheduler, admitted, replies


@pytest.mark.parametrize("params,prompt,config,error", [
    (dict(clover_mode="denoise"), 59, {}, "full preview block"),
    (dict(clover_mode="denoise", max_tokens=10), 49, {}, "full preview block"),
    (dict(clover_mode="denoise", clover_width=16), 8, dict(clover_max_width=8), "reserved maximum"),
    (dict(clover_mode="selfspec", clover_width=32), 8, {}, "W=8/16"),
    (dict(clover_mode="denoise", temperature=1), 8, {}, "greedy"),
    (dict(clover_mode="denoise"), 8, dict(max_running_req=2), "max_running_req=1"),
    (dict(clover_mode="wrong"), 8, {}, "Invalid request-level"),
    (dict(clover_width=0), 8, {}, "Invalid request-level"),
])
def test_invalid_clover_request_is_rejected_before_admission(params, prompt, config, error):
    from freetoken.message import ErrorReplyMsg, UserMsg

    scheduler, admitted, replies = admission_scheduler(**config)
    invalid = UserMsg(1, torch.zeros(prompt, dtype=torch.int32),
        SamplingParams(**{**dict(max_tokens=1), **params}))
    scheduler._process_one_msg(invalid)
    assert admitted == []
    assert len(replies) == 1 and isinstance(replies[0], ErrorReplyMsg)
    assert error in replies[0].error
    valid = UserMsg(2, torch.zeros(8, dtype=torch.int32), SamplingParams(max_tokens=1))
    scheduler._process_one_msg(valid)
    assert [msg.uid for msg in admitted] == [2]
    assert len(replies) == 1


def test_clover_admission_freezes_defaults_without_mutating_shared_params():
    from freetoken.message import UserMsg

    scheduler, admitted, replies = admission_scheduler(clover_mode="denoise", clover_width=16)
    params = SamplingParams(max_tokens=17)
    scheduler._process_one_msg(UserMsg(1, torch.zeros(32, dtype=torch.int32), params))
    assert not replies
    assert admitted[0].sampling_params.clover_mode == "denoise"
    assert admitted[0].sampling_params.clover_width == 16
    assert params.clover_mode is None and params.clover_width is None


@pytest.mark.parametrize("backend", ["naive", "cpu"])
def test_block_backend_is_checked_before_admission(backend):
    from freetoken.message import UserMsg

    scheduler, admitted, replies = admission_scheduler()
    if backend == "naive":
        scheduler.cache_manager.is_hybrid = False
    else:
        scheduler.engine.moe_offload_cache = SimpleNamespace(decode_target="cpu", cpu_layer_ids=())
    scheduler._process_one_msg(UserMsg(1, torch.zeros(8, dtype=torch.int32),
        SamplingParams(max_tokens=1, clover_mode="denoise")))
    assert not admitted and len(replies) == 1


@pytest.mark.parametrize("kwargs,error", [
    (dict(clover_mode="selfspec", clover_width=32, max_running_req=1), "W=8/16"),
    (dict(clover_mode="denoise", max_running_req=2), "max_running_req=1"),
])
def test_invalid_static_block_configuration_fails_before_model_loading(kwargs, error):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match=error):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, **kwargs)


def test_offline_generation_reports_rejection_and_finishes_valid_requests():
    from freetoken.llm.llm import LLM, RequestAllFinished, RequestStatus
    from freetoken.message import DetokenizeMsg, ErrorReplyMsg

    llm = LLM.__new__(LLM)
    llm.eos_token_ids = set()
    llm.tokenizer = SimpleNamespace(decode=lambda ids: str(ids))

    def run():
        llm.status_map = {i: RequestStatus(i, [7], []) for i in range(2)}
        llm.offline_send_result([ErrorReplyMsg(0, "invalid Clover request", "invalid_request")])
        llm.offline_send_result([DetokenizeMsg(1, 9, True)])
        raise RequestAllFinished

    llm.run_forever = run
    result = llm.generate([[7], [7]], SamplingParams(max_tokens=1))
    assert result[0]["error"] == "invalid Clover request"
    assert result[0]["error_code"] == "invalid_request"
    assert result[0]["token_ids"] == []
    assert result[1] == {"text": "[9]", "token_ids": [9]}


@pytest.mark.parametrize("mode,expected", [
    ("causal", ["schedule", "drain"]), ("selfspec", ["drain", "schedule"]),
    ("denoise", ["drain", "schedule"]),
])
def test_overlap_drains_only_block_requests_before_scheduling(mode, expected):
    scheduler, _, _ = admission_scheduler()
    scheduler._pending_rebuild = None
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.prefill_manager.runnable = False
    scheduler.receive_msg = lambda **kwargs: []
    scheduler.engine.stream = object()
    scheduler.stream = SimpleNamespace(wait_stream=lambda stream: None)
    scheduler._flush_abort_acks = lambda: None
    events = []
    scheduler._process_last_data = lambda data: events.append("drain") if data is not None else None
    scheduler._schedule_next_batch = lambda: events.append("schedule")
    batch = Batch([SimpleNamespace(sampling_params=SamplingParams(clover_mode=mode))], "prefill")
    scheduler.overlap_loop((SimpleNamespace(batch=batch), None))
    assert events == expected


def test_first_selfspec_window_sees_pending_causal_token_in_stop_history():
    scheduler, _, _ = admission_scheduler()
    scheduler._pending_rebuild = None
    scheduler.prefill_manager.runnable = False
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.receive_msg = lambda **kwargs: []
    scheduler.engine.stream = object()
    scheduler.stream = SimpleNamespace(wait_stream=lambda stream: None)
    scheduler._flush_abort_acks = lambda: None
    scheduler.eos_token_ids = set()
    scheduler.tokenizer = SimpleNamespace(decode=lambda ids: ''.join(chr(i) for i in ids))
    req = Req(torch.tensor([88], dtype=torch.int32), 0, 0, 10, 1,
        SamplingParams(clover_mode="selfspec", stop_strs=["AB"]), None)
    previous = (SimpleNamespace(batch=Batch([req], "prefill")), None)
    scheduler._process_last_data = lambda data: req.append_host(torch.tensor([65])) if data else None
    limits = []

    def schedule():
        limits.append(scheduler._mtp_delivery_limit(req, [66, 67]))

    scheduler._schedule_next_batch = schedule
    scheduler.overlap_loop(previous)
    assert limits == [(1, True)]


@pytest.mark.parametrize("abort_hook", [True, False])
def test_causal_batch_request_error_releases_every_request(abort_hook):
    from freetoken.engine.speculative import SpeculativeRequestError
    from freetoken.scheduler.scheduler import ForwardInput
    from tests.scheduler.test_abort_inflight_prefill import _launch_req

    runtime, scheduler, manager, tables, first, batch, replies = request_runtime()
    second = _launch_req(manager.linear_state_pool, manager, tables,
                         torch.tensor([7, 8, 9], dtype=torch.int32))
    second.uid = first.uid + 1
    batch.reqs.append(second)
    scheduler.decode_manager.running_reqs.add(second)
    runtime.eligible = lambda batch: False
    if not abort_hook:
        scheduler.engine.mtp_runtime = SimpleNamespace(eligible=lambda batch: False)
    def fail(batch, args):
        raise SpeculativeRequestError("causal history cannot be rebuilt", code="invalid_state")
    scheduler.engine.forward_batch = fail
    indices = (torch.tensor([first.table_idx, second.table_idx]), torch.tensor([2, 2]))
    assert Scheduler._forward(scheduler, ForwardInput(batch, None, indices, indices)) is None
    assert {reply.uid for reply in replies} == {first.uid, second.uid}
    assert all(reply.code == "invalid_state" for reply in replies)
    assert first.aborted and second.aborted
    assert first.table_idx == second.table_idx == -1
    assert not scheduler.decode_manager.running_reqs
    assert manager.available_size == manager.num_pages
    manager.check_integrity()


def test_abort_rounds_reservation_up_without_freeing_a_partially_owned_page():
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.table import TableManager
    from tests.scheduler.test_abort_inflight_prefill import _launch_req

    runtime, scheduler, old_manager, _, old_req, _, _ = request_runtime()
    scheduler._free_req_resources(old_req)
    pool = old_manager.linear_state_pool
    page_table = torch.zeros((4, 64), dtype=torch.int32)
    manager = CacheManager(16, 4, page_table, "hybrid_radix", linear_state_pool=pool)
    tables = TableManager(max_running_reqs=4, page_table=page_table)
    req = _launch_req(pool, manager, tables, torch.tensor([3, 4, 5], dtype=torch.int32))
    req.device_len, req.cached_len, req.mtp_reserved_len = 3, 2, 3
    req.max_device_len = 19
    first_page = page_table[req.table_idx, 0].item()
    runtime.engine.page_table = page_table
    scheduler.cache_manager, scheduler.table_manager = manager, tables
    runtime._attempts[req] = SimpleNamespace(manager=manager, reserved_len=3,
                                           device_len=3, clean_ready=False)
    manager.allocate_paged([req], reserve_tokens=8)
    runtime.abort_request(req)
    assert first_page not in manager.free_slots.tolist()
    assert torch.all(manager.free_slots % 4 == 0)
    assert manager.available_size == manager.num_pages * 4 - 4
    scheduler._free_req_resources(req)
    assert manager.available_size == manager.num_pages * 4
    manager.check_integrity()
