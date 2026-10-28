"""Accepted top-k integers must preserve unfiltered and mixed-batch semantics."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler


@pytest.mark.parametrize('top_k', [16, 17, 2**31 - 1, 2**31, 2**80])
def test_top_k_above_vocabulary_does_not_overflow_device_int32(top_k):
    """Prepare real device tensors; a full-vocabulary row needs no top-k filter."""
    sampler = Sampler(torch.device('cuda:0'), vocab_size=16)
    args = sampler.prepare(SimpleNamespace(reqs=[SimpleNamespace(
        sampling_params=SamplingParams(temperature=0.7, top_k=top_k),
    )]))
    assert args.top_k is None
    assert args.temperatures.tolist() == pytest.approx([0.7])
    assert args.greedy_mask is None


def test_mixed_top_k_preserves_bounded_filter_and_greedy_row():
    """Only the large row becomes unfiltered; greedy and bounded rows keep their roles."""
    sampler = Sampler(torch.device('cuda:0'), vocab_size=16)
    params = [SamplingParams(temperature=0.7, top_k=2**80),
              SamplingParams(temperature=0.7, top_k=3),
              SamplingParams(temperature=0, top_k=2**80)]
    args = sampler.prepare(SimpleNamespace(reqs=[SimpleNamespace(sampling_params=p) for p in params]))
    assert args.top_k.tolist() == [16, 3, 16]
    assert args.greedy_mask.tolist() == [False, False, True]
