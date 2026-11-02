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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 8, 16, 32])
@pytest.mark.parametrize("biases", [False, True])
def test_cuda_penalties_are_bitwise_equal_to_pinned_hf(width, biases):
    import os
    from pathlib import Path
    import runpy

    location = os.environ.get("CLOVER_HF_SOURCE")
    if not location:
        pytest.skip("set CLOVER_HF_SOURCE to the pinned local reference code")
    harness = runpy.run_path(
        str(Path(__file__).parents[2] / "benchmarks/clover_hf_reference.py")
    )
    sampling = harness["official_modules"](Path(location))[2]
    gen = SimpleNamespace(
        **{
            **PENALTY_DEFAULTS,
            "presence_penalty": 0.9,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.05,
            "dllm_length_penalty": 2.0 if biases else 0.0,
            "dllm_length_penalty_start": 510,
            "dllm_length_penalty_window": 8,
            "dllm_think_end_bias": 3.0 if biases else 0.0,
            "dllm_think_end_start": 510,
            "dllm_think_end_window": 8,
            "dllm_think_end_token_id": 4,
        }
    )
    native, reference = [
        kind(gen, 248832, "cuda", [0, 5])
        for kind in (CloverPenaltyState, sampling.CloverPenaltyState)
    ]
    torch.manual_seed(width)
    history = torch.randint(8, 8192, (512,), device="cuda")
    history[:4] = torch.tensor([8, 8, 9, 10], device="cuda")
    native.commit(history)
    reference.commit(history)
    logits = torch.randn(width, 248832, device="cuda")
    logits[:, 151669] = -float("inf")
    logits[:, 8:11] = torch.tensor([-0.0, 1e20, -1e20], device="cuda")
    block = torch.full((width,), 151669, device="cuda", dtype=torch.int32)
    expected = reference.apply(logits.clone(), block, 151669)
    actual = native.apply(logits.clone(), block, 151669)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if biases:
        block[-1] = 4
        expected = reference.apply(logits.clone(), block, 151669)
        actual = native.apply(logits.clone(), block, 151669)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [8, 16, 32])
def test_fused_pick_matches_torch_penalties_and_confidence(width):
    from types import SimpleNamespace
    from freetoken.models.clover_moe.generation import CloverPenaltyState, PENALTY_DEFAULTS
    from freetoken.kernel.triton.clover_pick import pick_with_confidence

    vocab, mask = 248320, 151669
    gen = SimpleNamespace(**{**PENALTY_DEFAULTS, "do_sample": False,
        "presence_penalty": .9, "frequency_penalty": .15, "repetition_penalty": 1.05,
        "dllm_length_penalty": 3., "dllm_length_penalty_start": 0, "dllm_length_penalty_window": 100,
        "dllm_think_end_bias": 2., "dllm_think_end_start": 0, "dllm_think_end_window": 100,
        "dllm_think_end_token_id": 123})
    state = CloverPenaltyState(gen, vocab, "cuda", [248046, 248044])
    state.commit(torch.tensor([9, 9, 111, 221], device="cuda"))
    block = torch.full((width,), mask, device="cuda", dtype=torch.int32)
    logits = torch.randn(width, vocab, device="cuda", dtype=torch.bfloat16)
    expected = logits.float().clone()
    expected[:, mask] = -float("inf")
    state.apply(expected, block, mask)
    picks, confidence = pick_with_confidence(logits, state, block, mask)
    assert torch.equal(picks.long(), expected.argmax(-1))
    torch.testing.assert_close(confidence, expected.softmax(-1).max(-1).values, rtol=2e-5, atol=1e-7)


def test_transaction_rejects_padding_and_incorrect_layer_set():
    from freetoken.models.clover_moe.preview import CloverTransactions

    pool = SimpleNamespace(device='cpu', recurrent_states=torch.zeros(2, 1, 2, 2),
                           conv_states=torch.zeros(2, 1, 2, 3))
    transaction = CloverTransactions(pool, 8)
    transaction.begin(8, 3)
    transaction.written.update((0, 2))
    with pytest.raises(ValueError, match='incomplete'):
        transaction.commit(pool, 2, 0)
    transaction.written = {0, 1}
    with pytest.raises(ValueError, match='out-of-range'):
        transaction.commit(pool, 4, 0)
    transaction.recurrent[:, 1].fill_(7)
    transaction.conv[:, 1].fill_(9)
    transaction.commit(pool, 2, 0)
    assert torch.all(pool.recurrent_states == 7) and torch.all(pool.conv_states == 9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_padded_verify_kv_never_writes_unowned_locations():
    from freetoken.kernel.triton.clover_store import store_verify_kv

    keys = torch.full((16, 2, 64), 17., device='cuda', dtype=torch.bfloat16)
    values = keys.clone()
    k = torch.randn(8, 2, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    locations = torch.tensor([3, 7, 11, -100, 1000000, -1, -1, -1], device='cuda', dtype=torch.int32)
    valid = torch.arange(8, device='cuda') < 3
    expected_k, expected_v = keys.clone(), values.clone()
    expected_k[locations[:3]] = k[:3]
    expected_v[locations[:3]] = v[:3]
    store_verify_kv(keys, values, locations, k, v, valid)
    assert torch.equal(keys, expected_k) and torch.equal(values, expected_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('width', [8, 16])
@pytest.mark.parametrize('rule', ['exact', 'threshold'])
@pytest.mark.parametrize('break_row', [None, 3])
def test_fused_verify_matches_sequential_hypothetical_history(width, rule, break_row):
    import copy
    from freetoken.kernel.triton.clover_verify import penalize_logits, verify_and_deliver

    vocab, mask = 257, 256
    gen = SimpleNamespace(**{**PENALTY_DEFAULTS, 'presence_penalty': .9,
        'frequency_penalty': .15, 'repetition_penalty': 1.05,
        'dllm_length_penalty': 3., 'dllm_length_penalty_start': 8192,
        'dllm_length_penalty_window': 10, 'dllm_think_end_bias': 2.,
        'dllm_think_end_start': 8192, 'dllm_think_end_window': 10,
        'dllm_think_end_token_id': 123})
    state = CloverPenaltyState(gen, vocab, 'cuda', [253, 254])
    state.commit(torch.tensor([7, 7, 18], device='cuda'))
    state.length = 8190
    inputs = torch.full((width,), 7, dtype=torch.int32, device='cuda')
    inputs[0] = 123
    logits = torch.zeros(width, vocab, device='cuda', dtype=torch.bfloat16)
    logits[:, 7] = 12
    if break_row is not None:
        logits[break_row, 7] = -10
        logits[break_row, 11] = 20
    reference = logits.float().clone()
    reference[:, mask] = -float('inf')
    hypothetical = copy.copy(state)
    hypothetical.counts = state.counts.clone()
    for row in range(width):
        hypothetical.apply(reference[row:row+1], inputs[row:row+1], mask)
        if row + 1 < width:
            hypothetical.commit(inputs[row+1:row+2].long())
    frozen = state.counts.clone()
    actual = penalize_logits(logits, state, mask, inputs)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    accepted = int(clover_correct_drafts(reference[None], inputs[None], accept_rule=rule).item())
    packet = verify_and_deliver(logits, inputs, state, mask, rule=rule).cpu().tolist()
    assert packet[0] == accepted
    assert packet[1:accepted+2] == inputs[1:accepted+1].tolist() + [int(reference[accepted].argmax())]
    assert torch.equal(state.counts, frozen) and state.length == 8190
    causal = logits[:1].float().clone()
    causal[:, mask] = -float('inf')
    state.apply(causal, torch.tensor([mask], device='cuda'), mask)
    torch.testing.assert_close(penalize_logits(logits[:1], state, mask), causal, rtol=0, atol=0)
