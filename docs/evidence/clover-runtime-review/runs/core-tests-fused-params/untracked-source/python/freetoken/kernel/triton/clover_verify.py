"""Response penalties and linear-chain verification without hypothetical state copies."""

import torch
import triton
import triton.language as tl


@triton.jit
def _adjust(value, token, row, COUNTS, INPUTS, LENGTH, MASK: tl.constexpr,
            V: tl.constexpr, ACTIVE: tl.constexpr, HYPOTHETICAL: tl.constexpr,
            PARAMS: tl.constexpr):
    PRESENCE: tl.constexpr = PARAMS[0]
    FREQUENCY: tl.constexpr = PARAMS[1]
    REPETITION: tl.constexpr = PARAMS[2]
    LENGTH_BIAS: tl.constexpr = PARAMS[3]
    LENGTH_START: tl.constexpr = PARAMS[4]
    LENGTH_WINDOW: tl.constexpr = PARAMS[5]
    THINK_BIAS: tl.constexpr = PARAMS[6]
    THINK_START: tl.constexpr = PARAMS[7]
    THINK_WINDOW: tl.constexpr = PARAMS[8]
    THINK_ID: tl.constexpr = PARAMS[9]
    STOP_IDS: tl.constexpr = PARAMS[10]
    value = tl.where(token == MASK, -float('inf'), value)
    if ACTIVE:
        count = tl.load(COUNTS + token, token < V, other=0)
        if HYPOTHETICAL:
            for index in range(1, row + 1):
                previous = tl.load(INPUTS + index)
                count += (token == previous).to(count.dtype)
        if FREQUENCY != 0.:
            value = value - count.to(tl.float32) * FREQUENCY
        if PRESENCE != 0.:
            value = value - (count > 0).to(tl.float32) * PRESENCE
        if REPETITION != 1.:
            scaled = tl.where(value < 0, value * REPETITION, tl.div_rn(value, REPETITION))
            value = tl.where(count > 0, scaled, value)
    position = LENGTH + row + 1.
    if LENGTH_BIAS != 0.:
        bias = tl.minimum(tl.maximum((position - LENGTH_START) / LENGTH_WINDOW, 0.), 1.) * LENGTH_BIAS
        for index in tl.static_range(len(STOP_IDS)):
            value = tl.where(token == STOP_IDS[index], value + bias, value)
    if THINK_BIAS != 0.:
        closed = tl.load(COUNTS + THINK_ID) > 0
        if HYPOTHETICAL:
            # Row zero sees the pending input; later rows also count their proposals.
            for index in range(1, row + 1):
                closed |= tl.load(INPUTS + index) == THINK_ID
            closed |= tl.load(INPUTS + row) == THINK_ID
        bias = tl.where(closed, 0.,
            tl.minimum(tl.maximum((position - THINK_START) / THINK_WINDOW, 0.), 1.) * THINK_BIAS)
        value = tl.where(token == THINK_ID, value + bias, value)
    return value


@triton.jit
def _penalize(LOGITS, OUT, COUNTS, INPUTS, STRIDE: tl.constexpr, V: tl.constexpr,
              LENGTH, MASK: tl.constexpr, ACTIVE: tl.constexpr,
              HYPOTHETICAL: tl.constexpr, PARAMS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    token = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(LOGITS + row * STRIDE + token, token < V, other=-float('inf')).to(tl.float32)
    value = _adjust(value, token, row, COUNTS, INPUTS, LENGTH, MASK, V,
                    ACTIVE, HYPOTHETICAL, PARAMS)
    tl.store(OUT + row * V + token, value, token < V)


@triton.jit
def _verify_rows(LOGITS, COUNTS, INPUTS, PICKS, MATCHES, STRIDE: tl.constexpr,
                 V: tl.constexpr, W: tl.constexpr, LENGTH, MASK: tl.constexpr,
                 ACTIVE: tl.constexpr, PARAMS: tl.constexpr, EXACT: tl.constexpr,
                 RATIO: tl.constexpr, DELTA: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, TILE)
    maximum = tl.full((TILE,), -float('inf'), tl.float32)
    winner = tl.full((TILE,), 2147483647, tl.int32)
    normalizer = tl.zeros((TILE,), tl.float32)
    for start in range(tl.cdiv(V, TILE)):
        token = start * TILE + lane
        value = tl.load(LOGITS + row * STRIDE + token, token < V, other=-float('inf')).to(tl.float32)
        value = _adjust(value, token, row, COUNTS, INPUTS, LENGTH, MASK, V, ACTIVE, True, PARAMS)
        new_max = tl.maximum(maximum, value)
        if not EXACT:
            normalizer = tl.where(new_max == -float('inf'), 0.,
                normalizer * tl.exp(maximum - new_max) + tl.exp(value - new_max))
        winner = tl.where((value > maximum) | ((value == maximum) & (token < winner)), token, winner)
        maximum = new_max
    row_max = tl.max(maximum, 0)
    pick = tl.min(tl.where(maximum == row_max, winner, 2147483647), 0)
    proposal = tl.load(INPUTS + row + 1, row < W - 1, other=0)
    if EXACT:
        match = pick == proposal
    else:
        denominator = tl.sum(normalizer * tl.exp(maximum - row_max), 0)
        entropy_parts = tl.zeros((TILE,), tl.float32)
        for start in range(tl.cdiv(V, TILE)):
            token = start * TILE + lane
            value = tl.load(LOGITS + row * STRIDE + token, token < V, other=-float('inf')).to(tl.float32)
            value = _adjust(value, token, row, COUNTS, INPUTS, LENGTH, MASK, V, ACTIVE, True, PARAMS)
            probability = tl.exp(value - row_max) / denominator
            entropy_parts -= probability * tl.log(tl.maximum(probability, 1.e-12))
        entropy = tl.sum(entropy_parts, 0)
        value = tl.load(LOGITS + row * STRIDE + proposal).to(tl.float32)
        value = _adjust(value, proposal, row, COUNTS, INPUTS, LENGTH, MASK, V, ACTIVE, True, PARAMS)
        probability = tl.exp(value - row_max) / denominator
        match = (probability >= tl.minimum(RATIO, DELTA * tl.exp(-entropy))) & (probability > 0)
    tl.store(PICKS + row, pick)
    tl.store(MATCHES + row, match & (row < W - 1))


@triton.jit
def _deliver(INPUTS, PICKS, MATCHES, PACKET, W: tl.constexpr, TILE: tl.constexpr):
    row = tl.arange(0, TILE)
    matches = tl.load(MATCHES + row, row < W - 1, other=0)
    accepted = tl.min(tl.where((row < W - 1) & ~matches, row, W - 1), 0)
    bonus = tl.load(PICKS + accepted)
    proposal = tl.load(INPUTS + row + 1, row < W - 1, other=0)
    tl.store(PACKET, accepted)
    tl.store(PACKET + 1 + row, tl.where(row < accepted, proposal, bonus), row < W)


def _parameters(state):
    g = state.gen
    return (g.presence_penalty, g.frequency_penalty, g.repetition_penalty,
        g.dllm_length_penalty, g.dllm_length_penalty_start, g.dllm_length_penalty_window,
        g.dllm_think_end_bias, g.dllm_think_end_start, g.dllm_think_end_window,
        g.dllm_think_end_token_id or 0, tuple(state.stop_ids))


def penalize_logits(logits, state, mask_id, inputs=None, *, output=None):
    rows, vocab = logits.shape
    if output is None:
        output = torch.empty((rows, vocab), dtype=torch.float32, device=logits.device)
    _penalize[(rows, triton.cdiv(vocab, 1024))](logits, output,
        state.counts if state.active else logits, inputs if inputs is not None else logits,
        logits.stride(0), vocab, state.length, mask_id, state.active, inputs is not None,
        _parameters(state), 1024, enable_fp_fusion=False)
    return output


def verify_and_deliver(logits, inputs, state, mask_id, *, rule, ratio=.9, delta=.6):
    if rule not in ('exact', 'threshold'):
        raise ValueError('Unknown Clover acceptance rule')
    width, vocab = logits.shape
    picks = torch.empty(width, dtype=torch.int32, device=logits.device)
    matches = torch.empty(width, dtype=torch.bool, device=logits.device)
    packet = torch.empty(width + 1, dtype=torch.int32, device=logits.device)
    _verify_rows[(width,)](logits, state.counts if state.active else logits, inputs,
        picks, matches, logits.stride(0), vocab, width, state.length, mask_id,
        state.active, _parameters(state), rule == 'exact', ratio, delta, 1024,
        num_warps=4, enable_fp_fusion=False)
    _deliver[(1,)](inputs, picks, matches, packet, width, triton.next_power_of_2(width))
    return packet
