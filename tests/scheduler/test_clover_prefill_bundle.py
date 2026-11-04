"""A first denoised block must be wholly on the host before prefix insertion."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
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
