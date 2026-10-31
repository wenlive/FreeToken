"""A larger evaluation cap must rerun truncated cases and preserve closed ones."""

import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def arrival_metrics():
    script = Path(__file__).parents[2] / "benchmarks/clover_metrics.py"
    return runpy.run_path(str(script))["arrival_metrics"]


def test_terminal_eos_in_last_burst_is_excluded(arrival_metrics):
    measured = arrival_metrics(2, [0.1, 0.2, 0.2])
    assert measured["decode_tokens_per_second"] == pytest.approx(10)
    assert measured["token_arrival_seconds"] == [0.1, 0.2]
    assert measured["terminal_event_count"] == 1


def test_causal_eos_delay_does_not_extend_visible_interval(arrival_metrics):
    assert arrival_metrics(2, [0.1, 0.2, 0.5])[
        "decode_tokens_per_second"
    ] == pytest.approx(10)


def test_single_visible_burst_has_no_steady_rate(arrival_metrics):
    measured = arrival_metrics(2, [0.1, 0.1, 0.1])
    assert measured["first_burst_tokens"] == 2
    assert measured["decode_tokens_per_second"] is None


def test_cap_without_eos_keeps_all_arrivals(arrival_metrics):
    measured = arrival_metrics(3, [0.1, 0.2, 0.3])
    assert measured["terminal_event_count"] == 0
    assert measured["decode_tokens_per_second"] == pytest.approx(10)


def test_eos_only_response_has_no_visible_ttft(arrival_metrics):
    measured = arrival_metrics(0, [0.1])
    assert measured["ttft_seconds"] is None
    assert measured["first_burst_tokens"] == 0


@pytest.mark.parametrize("count,arrivals", [(3, [0.1]), (1, [0.1, 0.2, 0.3])])
def test_mismatched_arrival_counts_are_rejected(arrival_metrics, count, arrivals):
    with pytest.raises(ValueError, match="Arrival count"):
        arrival_metrics(count, arrivals)


def test_saved_report_timing_correction_preserves_source():
    script = Path(__file__).parents[2] / "benchmarks/clover_metrics.py"
    correct = runpy.run_path(str(script))["corrected_timing_report"]
    report = dict(
        runs=[
            dict(
                prompt="Count",
                mode="denoise",
                width=8,
                warmup=False,
                output_tokens=2,
                token_arrival_seconds=[0.1, 0.2, 0.2],
                rounds=[dict(steps=[4, 4])],
            )
        ],
        summary=[
            dict(
                prompt="Count",
                mode="denoise",
                width=8,
                decode_tokens_per_second=20,
                end_to_end_tokens_per_second=8,
                tokens_per_preview=4,
            )
        ],
    )
    corrected = correct(report)
    assert corrected["summary"][0]["decode_tokens_per_second"] == pytest.approx(10)
    assert corrected["summary"][0]["visible_tokens_per_preview"] == 1
    assert corrected["summary"][0]["tokens_per_preview"] == 4
    assert corrected["summary"][0]["end_to_end_tokens_per_second"] == 8
    assert report["summary"][0]["decode_tokens_per_second"] == 20
    assert report["runs"][0]["token_arrival_seconds"] == [0.1, 0.2, 0.2]


@pytest.mark.parametrize("change_temperature", [False, True])
def test_cap_extension_reuses_only_eos_cases(tmp_path, monkeypatch, change_temperature):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sources.json").write_text("{}", encoding="utf-8")
    (data / "gsm8k-first50.jsonl").write_text(
        "\n".join(
            json.dumps(dict(question=q, answer=a))
            for q, a in (("first", "#### 3"), ("second", "#### 6"))
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    previous = dict(
        state="complete",
        backend="native",
        precision="nvfp4",
        checkpoint=str(checkpoint),
        protocol=dict(
            tokens=2,
            limit=2,
            selection="first N unshuffled",
            thinking=False,
            temperature=1 if change_temperature else 0,
            penalties="checkpoint defaults",
            grader="GSM final number; HumanEval restricted subprocess 5s",
        ),
        sources={},
        datasets="gsm8k",
        summary={},
        cases=[
            dict(
                dataset="gsm8k",
                index=0,
                task_id=None,
                prompt_ids=[10],
                token_ids=[1, 2],
                text="#### 2",
                correct=False,
            ),
            dict(
                dataset="gsm8k",
                index=1,
                task_id=None,
                prompt_ids=[11],
                token_ids=[6],
                text="#### 6",
                correct=True,
            ),
        ],
    )
    old = tmp_path / "old.json"
    old.write_text(json.dumps(previous), encoding="utf-8")
    original_bytes = old.read_bytes()
    calls = []

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            self.tokenizer = SimpleNamespace(
                apply_chat_template=lambda messages, **kw: [10]
                if messages[0]["content"].startswith("first")
                else [11]
            )
            self.engine = SimpleNamespace(shutdown=lambda: None)

        def generate(self, ids, params):
            calls.append((ids, params.max_tokens))
            return [dict(text="#### 3", token_ids=[1, 2, 3])]

    llm_module, core_module = ModuleType("freetoken.llm"), ModuleType("freetoken.core")
    llm_module.LLM = FakeLLM
    core_module.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "freetoken.llm", llm_module)
    monkeypatch.setitem(sys.modules, "freetoken.core", core_module)
    output = tmp_path / "extended"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_clover_quality.py",
            "--source",
            str(checkpoint),
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--output",
            str(output),
            "--backend",
            "native",
            "--datasets",
            "gsm8k",
            "--limit",
            "2",
            "--tokens",
            "4",
            "--extend-from",
            str(old),
        ],
    )
    script = Path(__file__).parents[2] / "benchmarks/eval_clover_quality.py"
    if change_temperature:
        with pytest.raises(ValueError, match="different protocol"):
            runpy.run_path(str(script), run_name="__main__")
        assert not calls
    else:
        runpy.run_path(str(script), run_name="__main__")
        result = json.loads((output / "results.json").read_text(encoding="utf-8"))
        assert calls == [([[10]], 4)]
        assert result["extension"]["reused_eos_cases"] == 1
        assert result["cases"][0]["extension_prefix_matches"]
        assert result["cases"][1] == previous["cases"][1]
        assert result["summary"]["gsm8k"] == dict(
            completed=2, correct=2, score=1.0, capped=0
        )
        assert result["state"] == "complete"
    assert old.read_bytes() == original_bytes
