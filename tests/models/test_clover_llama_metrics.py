"""Visible-token accounting for stock llama.cpp logs and SSE final packets."""

from pathlib import Path

import pytest


@pytest.mark.parametrize('native_tokens', [8, 12])
def test_denoise_budget_uses_visible_progress_and_keeps_hf_attainment_unclamped(native_tokens):
    import runpy

    analyze = runpy.run_path(str(Path(__file__).parents[2]/'benchmarks/analyze_clover_budget.py'))['analyze']
    trace = ([dict(phase='preview',cuda_ms=2.)]*4 +
             [dict(phase='verify',cuda_ms=2.)]*2 +
             [dict(phase='state_commit',cuda_ms=.5)]*2 +
             [dict(phase='save_ahead',cuda_ms=.5)]*2 +
             [dict(phase='round',cuda_ms=9.)]*2 + [dict(phase='advance',cuda_ms=2.)])
    row = dict(prompt='fixture',mode='denoise',width=8,warmup=False,
        output_tokens=native_tokens,runtime_breakdown=trace,decode_tokens_per_second=123.,
        end_to_end_tokens_per_second=100.,ttft_seconds=.01,first_burst_tokens=4)
    hf = dict(cases=[dict(prompt='fixture',width=8,visible_token_ids=list(range(8)),
                         token_ids=list(range(9)),blocks=[[4,4],[4,4]])])
    case = analyze(dict(runs=[row],settings={}),hf)['cases'][0]
    assert case['hf_useful_tokens_per_preview'] == 2
    assert case['hf_resolved_slots_per_preview'] == 4
    assert case['medians']['hf_conditioned_tokens_per_second'] == pytest.approx(8000/14)
    assert case['medians']['native_compute_efficiency'] == pytest.approx(14/16)
    assert case['medians']['generation_envelope_tokens_per_second'] == native_tokens*1000/16
    assert case['medians']['hf_conditioned_attainment'] == pytest.approx(native_tokens*14/(16*8))


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    import bench_clover_llama

    return bench_clover_llama


def test_cli_uses_visible_pieces_and_producer_time_not_eos_inclusive_counter(metrics):
    records = [
        dict(type="log", level="none", msg=text, time=at)
        for text, at in [
            ("\n\n", 0),
            ("prompt", 100),
            ("one", 200),
            (" two", 300),
            ("\n\n", 500),
            ("\n\n", 600),
        ]
    ]
    records.append(
        dict(
            type="log",
            level="info",
            msg="encoded 1 tokens in 0.001 seconds\ndecoded 4 tokens in 0.002 seconds",
        )
    )
    row = metrics.cli_metrics(records, dict(prompt_ids=[17], text="prompt"))
    assert row["output_tokens"] == 2 and row["builtin_predicted_tokens"] == 4
    assert row["unprinted_builtin_tokens"] == 2
    assert row["wall_seconds"] == pytest.approx(0.0004)
    assert row["ttft_seconds"] == pytest.approx(0.0001)


def test_cli_refuses_partial_timing_or_missing_output_boundary(metrics):
    with pytest.raises(ValueError, match="complete official CLI"):
        metrics.cli_metrics([], dict(prompt_ids=[], text=""))


def test_sse_deduplicates_terminal_full_list_and_excludes_eos(metrics):
    events = [
        dict(seconds=0.1, data=dict(tokens=[10, 11], content="first two")),
        dict(seconds=0.2, data=dict(tokens=[12], content=" third")),
        dict(
            seconds=0.3,
            data=dict(
                stop=True,
                tokens=[10, 11, 12, 248046],
                content="",
                timings=dict(predicted_n=4),
            ),
        ),
    ]
    row = metrics.stream_metrics(events, 0.3)
    assert row["token_ids"] == [10, 11, 12]
    assert row["all_token_ids"] == [10, 11, 12, 248046]
    assert row["token_arrival_seconds"] == [0.1, 0.1, 0.2]
    assert row["end_to_end_tokens_per_second"] == 10


def test_sse_appends_separate_eos_and_requires_terminal_packet(metrics):
    events = [dict(seconds=0.1, data=dict(tokens=[10], content="x"))]
    with pytest.raises(ValueError, match="terminal"):
        metrics.stream_metrics(events, 0.2)
    events.append(dict(seconds=0.2, data=dict(stop=True, tokens=[248044], content="")))
    row = metrics.stream_metrics(events, 0.2)
    assert row["token_ids"] == [10] and row["output_tokens"] == 1


def test_sse_uses_actual_terminal_eog_when_llama_has_additional_stop_ids(metrics):
    events = [
        dict(seconds=0.1, data=dict(tokens=[10], content="x")),
        dict(
            seconds=0.2, data=dict(stop=True, stop_type="eos", tokens=[999], content="")
        ),
    ]
    row = metrics.stream_metrics(events, 0.2)
    assert row["all_token_ids"] == [10, 999]
    assert row["token_ids"] == [10]
