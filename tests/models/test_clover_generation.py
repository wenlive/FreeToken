from types import SimpleNamespace

import pytest
import torch

from freetoken.models.clover_moe.acceptance import clover_correct_drafts
from freetoken.models.clover_moe.generation import CloverPenaltyState, PENALTY_DEFAULTS


def test_response_penalties_order_and_row_biases():
    gen = SimpleNamespace(
        **{
            **PENALTY_DEFAULTS,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.9,
            "repetition_penalty": 1.05,
            "dllm_length_penalty": 2.0,
            "dllm_length_penalty_start": 3,
            "dllm_length_penalty_window": 2,
            "dllm_think_end_bias": 3.0,
            "dllm_think_end_start": 0,
            "dllm_think_end_window": 4,
            "dllm_think_end_token_id": 4,
        }
    )
    state = CloverPenaltyState(gen, 7, "cpu", [0, 5])
    state.commit(torch.tensor([1, 1, 2]))
    logits = torch.tensor(
        [[1.0, 2.0, -1.0, 3.0, 0.0, -2.0, 4.0], [-1.0, -2.0, 1.0, 0.0, 2.0, 3.0, 4.0]]
    )
    expected = logits.clone()
    for row in range(2):
        for token, count in ((1, 2), (2, 1)):
            value = expected[row, token] - count * 0.2 - 0.9
            expected[row, token] = value * 1.05 if value < 0 else value / 1.05
        expected[row, [0, 5]] += row + 1
        expected[row, 4] += 3.0
    actual = state.apply(logits.clone(), torch.tensor([6, 6]), 6)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert state.length == 3 and state.counts.tolist() == [0, 2, 1, 0, 0, 0, 0]


def test_visible_or_committed_think_end_disables_bias():
    gen = SimpleNamespace(
        **{
            **PENALTY_DEFAULTS,
            "dllm_think_end_bias": 5.0,
            "dllm_think_end_start": 0,
            "dllm_think_end_window": 1,
            "dllm_think_end_token_id": 2,
        }
    )
    state = CloverPenaltyState(gen, 4, "cpu", [3])
    torch.testing.assert_close(
        state.apply(torch.zeros(2, 4), torch.tensor([2, 0]), 0), torch.zeros(2, 4)
    )
    state.commit(torch.tensor([2]))
    torch.testing.assert_close(
        state.apply(torch.zeros(2, 4), torch.tensor([0, 0]), 0), torch.zeros(2, 4)
    )


def test_exact_and_entropy_threshold_acceptance_have_distinct_semantics():
    probabilities = torch.tensor(
        [[[0.6, 0.3, 0.1], [0.8, 0.15, 0.05], [0.1, 0.8, 0.1]]]
    )
    candidates = torch.tensor([[2, 1, 1]])
    assert (
        clover_correct_drafts(
            probabilities.log(), candidates, accept_rule="exact"
        ).item()
        == 0
    )
    assert (
        clover_correct_drafts(
            probabilities.log(), candidates, accept_rule="threshold"
        ).item()
        == 1
    )
    blocked = probabilities.log()
    blocked[0, 0, 1] = -float("inf")
    assert clover_correct_drafts(blocked, candidates, 0.0, 0.0, "threshold").item() == 0


def test_verify_capacity_defaults_remain_four_and_can_be_explicitly_extended():
    from freetoken.attention.linear import build_fla_metadata

    req = SimpleNamespace(
        extend_len=16,
        cached_len=5,
        linear_slot_idx=1,
        table_idx=1,
        mamba_ping_pong=None,
    )
    batch = SimpleNamespace(padded_reqs=[req], is_decode=False, is_verify=True)
    with pytest.raises(ValueError, match="1-4 rows"):
        build_fla_metadata(batch, torch.device("cpu"))
    metadata = build_fla_metadata(batch, torch.device("cpu"), max_verify_rows=16)
    assert metadata.is_verify and metadata.cu_seqlens.tolist() == [0, 16]


def test_lowconfidence_strict_threshold_fallback_and_frozen_positions():
    from freetoken.models.clover_moe.generation import resolve_low_confidence

    state = CloverPenaltyState(
        SimpleNamespace(**PENALTY_DEFAULTS, do_sample=False), 4, "cpu", [0]
    )
    # Equal confidence is deliberately below the strict > gate. One argmax
    # fallback resolves position zero; later rounds must never overwrite it.
    first = torch.tensor([[2.0, 0.0, 0.0, -float("inf")]]).repeat(8, 1)
    threshold = float(first.softmax(-1)[0, 0])
    calls = []

    def preview(block):
        calls.append(block.clone())
        return (
            first.clone()
            if len(calls) == 1
            else torch.tensor([[-50.0, 50.0, -50.0, 0.0]]).repeat(8, 1)
        )

    block, steps = resolve_low_confidence(
        preview, state, width=8, mask_id=3, threshold=threshold, device="cpu"
    )
    assert steps == [1, 7] and block.tolist() == [0] + [1] * 7
    assert calls[0].tolist() == [3] * 8 and calls[1].tolist() == [0] + [3] * 7
    assert state.length == 0  # tentative tokens never enter the repetition history
