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


@pytest.mark.parametrize("mode", ["selfspec", "denoise"])
def test_tool_anchor_uses_the_emitted_tokens_absolute_position(mode):
    from freetoken.models.clover_moe.runtime import CloverRuntime

    runtime = object.__new__(CloverRuntime)
    pool = SimpleNamespace(
        conv_states=torch.zeros(2, 4, 1), recurrent_states=torch.zeros(2, 4, 1, 1)
    )
    runtime.engine = SimpleNamespace(
        config=SimpleNamespace(page_size=64), linear_state_pool=pool
    )
    runtime.transactions = SimpleNamespace(
        conv=torch.arange(32.0).reshape(2, 16, 1),
        recurrent=torch.arange(32.0).reshape(2, 16, 1, 1),
    )
    req = SimpleNamespace(
        mamba_ping_pong=[1, 2],
        mamba_next_track_idx=0,
        mamba_last_track_seqlen=None,
        toolcall_anchor_len=None,
    )
    # The anchor ends absolute prefix 64 in both cases. Selfspec has an input at
    # position 60; denoise starts emitting directly at position 60.
    anchor_index = 2 if mode == "selfspec" else 3
    delivered = [0] * (7 if mode == "selfspec" else 8)
    delivered[anchor_index] = 99
    kwargs = {} if mode == "selfspec" else {"emitted_start": 60}
    runtime._snapshot_boundary(
        req, 60, 7, delivered, SimpleNamespace(finished=False), 99, **kwargs
    )
    assert req.toolcall_anchor_len == req.mamba_last_track_seqlen == 64
    assert req.mamba_next_track_idx == 1
    torch.testing.assert_close(pool.conv_states[:, 1], runtime.transactions.conv[:, 3])
    torch.testing.assert_close(
        pool.recurrent_states[:, 1], runtime.transactions.recurrent[:, 3]
    )


def test_tool_anchor_snapshot_survives_later_block_boundaries():
    from freetoken.models.clover_moe.runtime import CloverRuntime

    runtime = object.__new__(CloverRuntime)
    pool = SimpleNamespace(
        conv_states=torch.full((2, 4, 1), 7.0),
        recurrent_states=torch.full((2, 4, 1, 1), 11.0),
    )
    runtime.engine = SimpleNamespace(
        config=SimpleNamespace(page_size=64), linear_state_pool=pool
    )
    runtime.transactions = SimpleNamespace(
        conv=torch.zeros(2, 16, 1), recurrent=torch.zeros(2, 16, 1, 1)
    )
    req = SimpleNamespace(
        mamba_ping_pong=[1, 2],
        mamba_next_track_idx=1,
        mamba_last_track_seqlen=64,
        toolcall_anchor_len=64,
    )
    runtime._snapshot_boundary(
        req, 124, 8, [0] * 8, SimpleNamespace(finished=False), 99
    )
    assert req.mamba_last_track_seqlen == 64 and req.mamba_next_track_idx == 1
    assert torch.all(pool.conv_states == 7) and torch.all(pool.recurrent_states == 11)
