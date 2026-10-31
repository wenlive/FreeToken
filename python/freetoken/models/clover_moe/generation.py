# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND. See the License for the specific language governing
# permissions and limitations under the License.
# Source: DiffuSpace/Clover-35B-A3B-Preview@4b0402b65e523d0183a14c431955438714cb16ff
# Paths: sampling_clover_moe.py; modeling_clover_moe.py (generate control flow)
# Local changes: native LowConfidence loop; bitwise-equivalent dense repetition
# on small CUDA rows avoids boolean-index synchronization. Reference operation
# order, length/thinking biases and confidence functions are retained.
"""Response-only penalties for Clover block LowConfidence generation.

Presence, frequency and repetition use committed response history.
Length and thinking controls provide gradual stopping biases.
All state belongs to one generate request; prompt tokens are never accumulated.
"""

import math

import torch


_DENSE_REPETITION = True  # benchmark-only opt-out; keep W32's smaller workspace


PENALTY_DEFAULTS = {
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "dllm_length_penalty": 0.0,
    "dllm_length_penalty_start": 8192,
    "dllm_length_penalty_window": 8192,
    "dllm_think_end_bias": 0.0,
    "dllm_think_end_start": 4096,
    "dllm_think_end_window": 4096,
    "dllm_think_end_token_id": None,
    "stop_token_ids": [],
}


def validate_penalties(gen, vocab_size):
    for name, default in PENALTY_DEFAULTS.items():
        if getattr(gen, name, None) is None:
            setattr(gen, name, default)
    for name in ("presence_penalty", "frequency_penalty"):
        value = getattr(gen, name)
        if not math.isfinite(value) or not -2 <= value <= 2:
            raise ValueError(f"{name} must be finite and in [-2, 2]")
    if not math.isfinite(gen.repetition_penalty) or not 0 < gen.repetition_penalty <= 2:
        raise ValueError("repetition_penalty must be finite and in (0, 2]")
    for prefix, strength in (
        ("dllm_length_penalty", "dllm_length_penalty"),
        ("dllm_think_end", "dllm_think_end_bias"),
    ):
        value = getattr(gen, strength)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{strength} must be finite and nonnegative")
        for suffix, minimum in (("start", 0), ("window", 1)):
            name = f"{prefix}_{suffix}"
            value = getattr(gen, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
    close = gen.dllm_think_end_token_id
    if close is not None and (
        not isinstance(close, int) or not 0 <= close < vocab_size
    ):
        raise ValueError("dllm_think_end_token_id must be a valid token ID")
    if gen.dllm_think_end_bias and close is None:
        raise ValueError("dllm_think_end_bias requires dllm_think_end_token_id")
    if not isinstance(gen.stop_token_ids, (list, tuple)) or any(
        not isinstance(i, int) or not 0 <= i < vocab_size for i in gen.stop_token_ids
    ):
        raise ValueError("stop_token_ids must be a list of valid token IDs")


class CloverPenaltyState:
    def __init__(self, gen, vocab_size, device, stop_ids):
        self.gen = gen
        self.length = 0
        self.active = bool(
            gen.presence_penalty
            or gen.frequency_penalty
            or gen.repetition_penalty != 1
            or gen.dllm_length_penalty
            or gen.dllm_think_end_bias
        )
        self.counts = (
            torch.zeros(vocab_size, dtype=torch.int64, device=device)
            if self.active
            else None
        )
        self.stop_ids = sorted({i for i in stop_ids if 0 <= i < vocab_size})

    def commit(self, tokens):
        """Count each returned response token once, after full block resolution."""
        self.length += tokens.numel()
        if self.active:
            self.counts.index_add_(
                0, tokens, torch.ones_like(tokens, dtype=self.counts.dtype)
            )

    def apply(self, logits, block, mask_id):
        """Mutate fresh [block, vocab] logits before sampling and confidence."""
        if not self.active:
            return logits
        gen = self.gen
        # Uncommitted block tokens must not penalize themselves or other slots.
        # This snapshot changes only when the returned block is committed.
        seen = self.counts > 0
        # Keep SGLang's additive operation order and FP32 rounding boundaries.
        if gen.frequency_penalty:
            logits.sub_(self.counts.float() * gen.frequency_penalty)
        if gen.presence_penalty:
            logits.sub_(seen * gen.presence_penalty)
        if gen.repetition_penalty != 1:
            if (
                _DENSE_REPETITION
                and logits.is_cuda
                and logits.dtype == torch.float32
                and logits.shape[0] <= 16
            ):
                # Boolean indexing synchronizes to size its nonzero result.
                # Full-row arithmetic keeps the same scalar operations on GPU.
                scaled = torch.where(
                    logits < 0,
                    logits * gen.repetition_penalty,
                    logits / gen.repetition_penalty,
                )
                logits.copy_(torch.where(seen[None, :], scaled, logits))
            else:
                values = logits[:, seen]
                logits[:, seen] = torch.where(
                    values < 0,
                    values * gen.repetition_penalty,
                    values / gen.repetition_penalty,
                )
        positions = self.length + torch.arange(
            1, logits.shape[0] + 1, device=logits.device
        )

        def ramp(start, window, strength):
            return ((positions.float() - start) / window).clamp(0, 1) * strength

        if gen.dllm_length_penalty and self.stop_ids:
            logits[:, self.stop_ids] += ramp(
                gen.dllm_length_penalty_start,
                gen.dllm_length_penalty_window,
                gen.dllm_length_penalty,
            )[:, None]
        if gen.dllm_think_end_bias:
            close = gen.dllm_think_end_token_id
            # In LowConfidence the whole block receives the bias until a close
            # token is visible or committed.
            visible_close = ((block != mask_id) & (block == close)).any()
            enabled = ~(seen[close] | visible_close)
            logits[:, close] += enabled * ramp(
                gen.dllm_think_end_start,
                gen.dllm_think_end_window,
                gen.dllm_think_end_bias,
            )
        return logits


def predict_with_confidence(logits, gen):
    """Sample with configured filters; confidence always uses T=1 logits."""
    if gen.do_sample:
        scaled = logits / gen.temperature
        if gen.top_k and 0 < gen.top_k < scaled.shape[-1]:
            cutoff = scaled.topk(gen.top_k, -1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < cutoff, -float("inf"))
        if gen.top_p is not None and gen.top_p < 1:
            sorted_logits, order = scaled.sort(-1, descending=True)
            probs = sorted_logits.softmax(-1)
            remove = (probs.cumsum(-1) - probs) > gen.top_p
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            scaled = torch.empty_like(scaled).scatter(-1, order, sorted_logits)
        prediction = torch.multinomial(scaled.softmax(-1), 1).squeeze(-1)
    else:
        prediction = logits.argmax(-1)
    confidence = logits.softmax(-1).gather(-1, prediction[:, None]).squeeze(-1)
    return prediction, confidence


def resolve_low_confidence(preview, state, *, width, mask_id, threshold, device):
    """Official full-width loop, including the final partial returned block.

    `preview` must be read-only. Only the caller may commit the resolved prefix.
    Confidence and selection remain on device; the small step trace is evidence.
    """
    block = torch.full((width,), mask_id, device=device, dtype=torch.int32)
    pending = torch.ones(width, device=device, dtype=torch.bool)
    steps = []
    for _ in range(width):
        logits = preview(block).float().clone()
        logits[:, mask_id] = -float("inf")
        state.apply(logits, block, mask_id)
        prediction, confidence = predict_with_confidence(logits, state.gen)
        confidence = confidence.masked_fill(~pending, -float("inf"))
        transfer = (confidence > threshold) & pending
        if not bool(transfer.any()):
            transfer[confidence.argmax()] = True
        block[transfer] = prediction[transfer].to(block.dtype)
        pending[transfer] = False
        steps.append(int(transfer.sum().item()))
        if not bool(pending.any()):
            return block, steps
    raise RuntimeError("LowConfidence failed to resolve its full block")
