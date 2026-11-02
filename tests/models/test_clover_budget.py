"""Analytical useful-token accounting, independent of model or GPU execution."""
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize('native_tokens', [8, 12])
def test_denoise_budget_excludes_discarded_slots_and_keeps_attainment_unclamped(native_tokens):
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
