"""Token timing must follow the callback bound by SchedulerIOMixin at construction."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from freetoken.message import DetokenizeMsg


def test_bound_offline_sender_is_timed_without_losing_delivery():
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_nvfp4_lora.py"
    spec = importlib.util.spec_from_file_location("nvfp4_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class OfflineLLM:
        def __init__(self):
            self.received = []
            self.send_result = self.offline_send_result

        def offline_send_result(self, messages):
            self.received.extend(messages)

        def generate(self, prompts, sampling):
            self.send_result([DetokenizeMsg(uid=0, next_token=17, finished=False)])
            self.send_result([DetokenizeMsg(uid=0, next_token=248046, finished=True)])
            return [{"text": "2", "token_ids": [17]}]

    llm = OfflineLLM()
    clock = benchmark.TokenClock(llm)
    with patch("torch.cuda.synchronize"), patch.object(
        benchmark.time, "perf_counter", side_effect=[100., 100.2, 100.4, 100.5]
    ):
        result = clock.generate(llm, [1, 2, 3], None)
    assert [message.next_token for message in llm.received] == [17, 248046]
    assert result["token_ids"] == [17]
    assert result["emitted_steps_including_eos"] == 2
    assert result["ttft_seconds"] == pytest.approx(.2)
    assert result["decode_steps_per_second"] == pytest.approx(5.)
    assert result["wall_seconds"] == pytest.approx(.5)
    assert result["output_tokens_per_wall_second"] == pytest.approx(2.)
