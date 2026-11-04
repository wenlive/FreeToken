"""Fused Clover greedy penalties, mask, argmax and normalized confidence.

Semantics: DiffuSpace/Clover-35B-A3B-Preview@4b0402b65e523d0183a14c431955438714cb16ff,
sampling_clover_moe.py (Apache-2.0). Independent tiled online-softmax implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pick(LOGITS, COUNTS, BLOCK_IDS, PICKS, CONFIDENCE, STRIDE: tl.constexpr,
          V: tl.constexpr, W: tl.constexpr, LENGTH, MASK: tl.constexpr,
          PRESENCE: tl.constexpr, FREQUENCY: tl.constexpr, REPETITION: tl.constexpr,
          LENGTH_BIAS: tl.constexpr, LENGTH_START: tl.constexpr, LENGTH_WINDOW: tl.constexpr,
          THINK_BIAS: tl.constexpr, THINK_START: tl.constexpr, THINK_WINDOW: tl.constexpr,
          THINK_ID: tl.constexpr, STOP_IDS: tl.constexpr, ACTIVE: tl.constexpr,
          TILE: tl.constexpr, BLOCK_W: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, TILE)
    maximum = tl.full((TILE,), -float("inf"), tl.float32)
    normalizer = tl.zeros((TILE,), tl.float32)
    winner = tl.full((TILE,), 2147483647, tl.int32)
    length_bias = tl.minimum(tl.maximum((LENGTH + row + 1. - LENGTH_START) / LENGTH_WINDOW, 0.), 1.) * LENGTH_BIAS
    if THINK_BIAS != 0.:
        w = tl.arange(0, BLOCK_W)
        block = tl.load(BLOCK_IDS + w, w < W, other=MASK)
        visible = tl.sum(((block == THINK_ID) & (w < W)).to(tl.int32), 0) > 0
        seen_close = tl.load(COUNTS + THINK_ID) > 0
        think_bias = tl.where(visible | seen_close, 0.,
            tl.minimum(tl.maximum((LENGTH + row + 1. - THINK_START) / THINK_WINDOW, 0.), 1.) * THINK_BIAS)
    for start in range(tl.cdiv(V, TILE)):
        token = start * TILE + lane
        value = tl.load(LOGITS + row * STRIDE + token, token < V, other=-float("inf")).to(tl.float32)
        value = tl.where(token == MASK, -float("inf"), value)
        if ACTIVE:
            count = tl.load(COUNTS + token, token < V, 0)
            if FREQUENCY != 0.:
                value = value - count.to(tl.float32) * FREQUENCY
            if PRESENCE != 0.:
                value = value - (count > 0).to(tl.float32) * PRESENCE
            if REPETITION != 1.:
                adjusted = tl.where(value < 0., value * REPETITION, tl.div_rn(value, REPETITION))
                value = tl.where(count > 0, adjusted, value)
        if LENGTH_BIAS != 0.:
            for stop in tl.static_range(len(STOP_IDS)):
                value = tl.where(token == STOP_IDS[stop], value + length_bias, value)
        if THINK_BIAS != 0.:
            value = tl.where(token == THINK_ID, value + think_bias, value)
        new_max = tl.maximum(maximum, value)
        normalizer = tl.where(new_max == -float("inf"), 0.,
            normalizer * tl.exp(maximum - new_max) + tl.exp(value - new_max))
        winner = tl.where((value > maximum) | ((value == maximum) & (token < winner)), token, winner)
        maximum = new_max
    row_max = tl.max(maximum, 0)
    denominator = tl.sum(normalizer * tl.exp(maximum - row_max), 0)
    token = tl.min(tl.where(maximum == row_max, winner, 2147483647), 0)
    tl.store(PICKS + row, token)
    tl.store(CONFIDENCE + row, 1. / denominator)


@triton.jit
def _resolve(BLOCK_IDS, PENDING, PICKS, CONFIDENCE, RESULT, W: tl.constexpr,
             THRESHOLD: tl.constexpr, TILE: tl.constexpr):
    row = tl.arange(0, TILE)
    pending = tl.load(PENDING + row, row < W, other=0)
    confidence = tl.load(CONFIDENCE + row, row < W, other=-float("inf"))
    confidence = tl.where(pending, confidence, -float("inf"))
    transfer = pending & (confidence > THRESHOLD)
    count = tl.sum(transfer.to(tl.int32), 0)
    best = tl.max(confidence, 0)
    fallback = tl.min(tl.where((confidence == best) & pending, row, 2147483647), 0)
    transfer = transfer | (pending & (count == 0) & (row == fallback))
    block = tl.load(BLOCK_IDS + row, row < W, other=0)
    pick = tl.load(PICKS + row, row < W, other=0)
    tl.store(BLOCK_IDS + row, tl.where(transfer, pick, block), row < W)
    tl.store(PENDING + row, pending & ~transfer, row < W)
    tl.store(RESULT, tl.sum(transfer.to(tl.int32), 0))
    tl.store(RESULT + 1, tl.sum((pending & ~transfer).to(tl.int32), 0))


def pick_with_confidence(logits, state, block, mask_id):
    gen = state.gen
    rows = logits.shape[0]
    picks = torch.empty(rows, dtype=torch.int32, device=logits.device)
    confidence = torch.empty(rows, dtype=torch.float32, device=logits.device)
    _pick[(rows,)](logits, state.counts if state.active else logits, block, picks, confidence,
        logits.stride(0), logits.shape[-1], block.numel(), state.length, mask_id,
        gen.presence_penalty, gen.frequency_penalty, gen.repetition_penalty,
        gen.dllm_length_penalty, gen.dllm_length_penalty_start, gen.dllm_length_penalty_window,
        gen.dllm_think_end_bias, gen.dllm_think_end_start, gen.dllm_think_end_window,
        gen.dllm_think_end_token_id or 0, tuple(state.stop_ids), state.active,
        2048, triton.next_power_of_2(block.numel()), num_warps=8, enable_fp_fusion=False)
    return picks, confidence


def resolve_step(block, pending, picks, confidence, threshold, result):
    _resolve[(1,)](block, pending, picks, confidence, result, block.numel(), threshold,
                   triton.next_power_of_2(block.numel()))
