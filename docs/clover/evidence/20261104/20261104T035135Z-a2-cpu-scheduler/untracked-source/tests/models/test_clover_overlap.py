"""Pure causal counts follow consumed device inputs despite delayed host delivery."""
from types import SimpleNamespace
import weakref

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.models.clover_moe.generation import CloverPenaltyState, PENALTY_DEFAULTS
from freetoken.models.clover_moe.runtime import CloverRuntime


def runtime_for_counts(device="cpu"):
    runtime = CloverRuntime.__new__(CloverRuntime)
    runtime.engine = SimpleNamespace(device=device, config=SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=17), clover_mode="causal",
        clover_width=8, clover_max_width=32, clover_threshold=.95))
    runtime.generation = SimpleNamespace(**{**PENALTY_DEFAULTS,
        "presence_penalty": .9, "repetition_penalty": 1.05,
        "frequency_penalty": .2, "dllm_think_end_bias": 2.,
        "dllm_think_end_token_id": 7, "dllm_think_end_start": 0,
        "dllm_think_end_window": 4})
    runtime.states = weakref.WeakKeyDictionary()
    runtime.mask_id, runtime.stop_ids = 16, [15]
    return runtime


def install_cpu_penalties(monkeypatch):
    from freetoken.kernel.triton import clover_verify

    def penalize(logits, state, mask_id, *, output):
        output.copy_(logits.float())
        output[:, mask_id] = -float("inf")
        return state.apply(output, torch.tensor([mask_id]), mask_id)

    monkeypatch.setattr(clover_verify, "penalize_logits", penalize)


@pytest.mark.parametrize("host_lag", [0, 1])
def test_causal_counts_and_penalties_follow_device_when_host_lags(monkeypatch, host_lag):
    install_cpu_penalties(monkeypatch)
    runtime = runtime_for_counts()
    req = Req(torch.tensor([4, 4], dtype=torch.int32), 0, 0, 12, 1,
        SamplingParams(clover_mode="causal"), None)
    logits = torch.arange(17, dtype=torch.float32)[None] / 5 - 2
    prefill = Batch([req], "prefill")
    req.complete_one()
    runtime.process_logits(prefill, logits)
    forced = [4, 4, 7, 6]
    oracle = CloverPenaltyState(runtime.generation, 17, "cpu", runtime.stop_ids)
    for index, token in enumerate(forced):
        history_length = index + 1 - host_lag
        while len(req.input_ids) - req.prompt_len < history_length:
            offset = len(req.input_ids) - req.prompt_len
            req.append_host(torch.tensor([forced[offset]], dtype=torch.int32))
        batch = Batch([req], "decode")
        batch.input_ids = torch.tensor([token], dtype=torch.int32)
        req.complete_one()
        actual = runtime.process_logits(batch, logits)
        oracle.commit(torch.tensor([token]))
        expected = logits.clone()
        expected[:, runtime.mask_id] = -float("inf")
        oracle.apply(expected, torch.tensor([runtime.mask_id]), runtime.mask_id)
        assert runtime.states[req].length == oracle.length
        torch.testing.assert_close(runtime.states[req].counts, oracle.counts, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        again = runtime.process_logits(batch, logits)
        torch.testing.assert_close(again, expected, rtol=0, atol=0)


def test_causal_counts_ignore_prompt_chunks_and_isolate_reused_tables(monkeypatch):
    from freetoken.scheduler.prefill import ChunkedReq

    install_cpu_penalties(monkeypatch)
    runtime = runtime_for_counts()
    chunk = ChunkedReq(torch.tensor([4, 4], dtype=torch.int32), 0, 0, 8, 1,
        SamplingParams(clover_mode="causal"), None, prompt_len=6)
    chunk.complete_one()
    runtime.process_logits(Batch([chunk], "prefill"), torch.zeros(1, 17))
    assert runtime.states[chunk].length == 0 and not runtime.states[chunk].counts.any()
    requests = [Req(torch.tensor([4] * 6, dtype=torch.int32), 0, 0, 8, uid,
        SamplingParams(clover_mode="causal"), None) for uid in (1, 2)]
    for req, token in zip(requests, (7, 8)):
        req.complete_one()
        runtime.process_logits(Batch([req], "prefill"), torch.zeros(1, 17))
        req.complete_one()
        batch = Batch([req], "decode")
        batch.input_ids = torch.tensor([token], dtype=torch.int32)
        runtime.process_logits(batch, torch.zeros(1, 17))
    assert runtime.states[requests[0]].counts[7] == 1
    assert runtime.states[requests[0]].counts[8] == 0
    assert runtime.states[requests[1]].counts[7] == 0
    assert runtime.states[requests[1]].counts[8] == 1


@pytest.mark.parametrize("modes,phase,drain", [
    (["causal"], "prefill", False), (["causal", "causal"], "decode", False),
    (["selfspec"], "prefill", True), (["selfspec"], "decode", True),
    (["denoise"], "prefill", True), (["causal", "selfspec"], "decode", True),
])
def test_drain_uses_request_mode_even_before_first_bundle(modes, phase, drain):
    runtime = runtime_for_counts()
    requests = [SimpleNamespace(sampling_params=SamplingParams(clover_mode=mode)) for mode in modes]
    assert runtime.requires_drain(Batch(requests, phase)) is drain


def test_mtp_keeps_its_existing_drain_barrier():
    from freetoken.models.qwen4_exp.mtp_runtime import MtpRuntime

    runtime = MtpRuntime.__new__(MtpRuntime)
    assert runtime.requires_drain(Batch([], "prefill"))
    assert runtime.requires_drain(Batch([], "decode"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_causal_logits_and_counts_match_forced_history_with_host_lag():
    runtime = runtime_for_counts("cuda")
    reqs = [Req(torch.tensor([4, 4], dtype=torch.int32), uid, 0, 12, uid,
        SamplingParams(clover_mode="causal"), None) for uid in range(2)]
    oracles = [CloverPenaltyState(runtime.generation, 17, "cpu", runtime.stop_ids) for _ in reqs]
    logits = (torch.arange(34, dtype=torch.float32).reshape(2, 17) / 5 - 2).bfloat16().cuda()
    for req in reqs:
        req.complete_one()
    runtime.process_logits(Batch(reqs, "prefill"), logits)
    for tokens in ((4, 8), (4, 8), (7, 4), (6, 7)):
        batch = Batch(reqs, "decode")
        batch.input_ids = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        for req in reqs:
            req.complete_one()
        actual = runtime.process_logits(batch, logits).cpu()
        for row, (req, oracle, token) in enumerate(zip(reqs, oracles, tokens)):
            oracle.commit(torch.tensor([token]))
            expected = logits[row:row + 1].float().cpu()
            expected[:, runtime.mask_id] = -float("inf")
            oracle.apply(expected, torch.tensor([runtime.mask_id]), runtime.mask_id)
            torch.testing.assert_close(actual[row:row + 1], expected, rtol=0, atol=0)
            torch.testing.assert_close(runtime.states[req].counts.cpu(), oracle.counts, rtol=0, atol=0)
