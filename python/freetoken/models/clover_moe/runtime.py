"""Single-model Clover generation, using the scheduler's verified-token bundles."""

import copy
from collections import deque
import time
from types import SimpleNamespace
import weakref

import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch
from .acceptance import clover_correct_drafts
from .generation import (
    CloverPenaltyState,
    PENALTY_DEFAULTS,
    validate_penalties,
    resolve_low_confidence,
    load_generation_settings,
)
from .preview import CloverTransactions, make_preview_batch


class CloverRuntime:
    def __init__(self, engine):
        self.engine = engine
        self.mask_id, raw = load_generation_settings(
            engine.config.model_path, engine.config.hf_config
        )
        if not 0 <= self.mask_id < engine.config.model_config.vocab_size:
            raise ValueError("Clover mask token is outside the model vocabulary")
        self.generation = SimpleNamespace(**{**PENALTY_DEFAULTS, **raw})
        if engine.config.clover_disable_penalties:
            for name in (
                "presence_penalty",
                "frequency_penalty",
                "dllm_length_penalty",
                "dllm_think_end_bias",
            ):
                setattr(self.generation, name, 0.0)
            self.generation.repetition_penalty = 1.0
        validate_penalties(self.generation, engine.config.model_config.vocab_size)
        eos = raw.get("eos_token_id", [])
        self.stop_ids = ([eos] if isinstance(eos, int) else list(eos)) + [
            raw.get("pad_token_id", 248044)
        ]
        self.states = weakref.WeakKeyDictionary()
        self.transactions = None
        self.rounds = deque(maxlen=4096)

    def prepare_graphs(self):
        from .graph import CloverBlockGraphs

        self.transactions = None
        self.states.clear()
        self.block_graphs = CloverBlockGraphs(self.engine)

    def close_graphs(self):
        if getattr(self, "block_graphs", None) is not None:
            self.block_graphs.close()
            self.block_graphs = None
        self.transactions = None
        self.states.clear()

    def before_native(self, batch):
        if not batch.is_prefill or self.transactions is None or self.transactions.capacity != 32:
            return None
        sparse_limit = self.engine.model.model.layers.op_list[0].mlp.sparse_prefill_limit
        if len(batch.input_ids) > sparse_limit:
            # A previous W32 window owns ~2 GiB. Release it before a new prompt's
            # full-layer prefill workspace, then allocate lazily at verification.
            self.block_graphs.clear_verify()
            self.transactions = None
        return None

    def _blocks(self):
        if getattr(self, "block_graphs", None) is None:
            from .graph import CloverBlockGraphs

            self.block_graphs = CloverBlockGraphs(self.engine)
        return self.block_graphs

    def _preview(self, req, tokens):
        return self._blocks().run(req, tokens, preview=True,
            coreset=self.engine.config.clover_coreset_top_p)

    def _verify(self, req, tokens, capacity):
        if self.transactions is None or self.transactions.capacity < capacity:
            self._blocks().clear_verify()
            self.transactions = None
            self.transactions = CloverTransactions(self.engine.linear_state_pool, capacity)
        return self._blocks().run(req, tokens, preview=False, transaction=self.transactions)

    def after_native(self, batch, native_state):
        pass

    def settings(self, req):
        params, config = req.sampling_params, self.engine.config
        mode = params.clover_mode or config.clover_mode
        width = params.clover_width or config.clover_width
        threshold = (
            config.clover_threshold
            if params.clover_threshold is None
            else params.clover_threshold
        )
        if (
            mode not in ("causal", "selfspec", "denoise")
            or width not in (8, 16, 32)
            or not 0 < threshold < 1
        ):
            raise ValueError("Invalid request-level Clover mode, width or threshold")
        if mode == "selfspec" and width == 32:
            raise ValueError("Clover selfspec currently supports W=8/16")
        return mode, width, threshold

    def state(self, req):
        state = self.states.get(req)
        history = req.input_ids[req.prompt_len :]
        if state is None or state.length > history.numel():
            state = CloverPenaltyState(
                self.generation,
                self.engine.config.model_config.vocab_size,
                self.engine.device,
                self.stop_ids,
            )
            self.states[req] = state
        if state.length < history.numel():
            state.commit(
                history[state.length :].to(device=self.engine.device, dtype=torch.int64)
            )
        return state

    def process_logits(self, batch, logits):
        scores = logits.float().clone()
        for row, req in enumerate(batch.reqs):
            mode, _, _ = self.settings(req)
            state = self.state(req)
            scores[row, self.mask_id] = -float("inf")
            state.apply(
                scores[row : row + 1],
                torch.full((1,), self.mask_id, device=scores.device),
                self.mask_id,
            )
        return scores

    def eligible(self, batch):
        if len(batch.reqs) != 1:
            if any(self.settings(req)[0] != "causal" for req in batch.reqs):
                raise ValueError(
                    "Clover block generation currently requires one running request"
                )
            return False
        req = batch.reqs[0]
        mode, width, _ = self.settings(req)
        if mode == "denoise":
            # Intermediate prompt chunks are state-only native forwards. The
            # final prompt chunk enters denoising before any causal token samples.
            if not req.can_decode:
                return False
            if not req.sampling_params.is_greedy:
                raise ValueError(
                    "Clover LowConfidence currently supports greedy sampling"
                )
            prefix = req.device_len
            if prefix + width > self.engine.max_seq_len:
                raise ValueError(
                    "Denoising requires room for a full preview block, including the final partial block"
                )
            return True
        if mode != "selfspec" or not batch.is_decode:
            return False
        if not req.sampling_params.is_greedy:
            raise ValueError(
                "Clover selfspec currently requires greedy boundary sampling"
            )
        return (
            req.extend_len == 1
            and req.cached_len > 0
            and req.remain_len >= width
            and req.cached_len + width <= self.engine.max_seq_len
        )

    def _verify_batch(self, request, tokens, *, capacity=16):
        req = copy.copy(request)
        req.device_len = req.cached_len + tokens.numel()
        req.mamba_ping_pong = None
        batch = Batch([req], "verify")
        batch.padded_reqs = [req]
        batch.input_ids = tokens
        batch.positions = torch.arange(
            req.cached_len, req.device_len, device=self.engine.device, dtype=torch.int32
        )
        batch.out_loc = self.engine.page_table[
            req.table_idx, req.cached_len : req.device_len
        ]
        batch.linear_table_idx = torch.tensor(
            [req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx],
            device=self.engine.device,
            dtype=torch.int32,
        )
        batch.fla_metadata = build_fla_metadata(
            batch, self.engine.device, max_verify_rows=len(tokens)
        )
        self.engine.attn_backend.prepare_metadata(batch)
        if self.transactions is None or self.transactions.capacity < capacity:
            # Free the old window before allocating a larger one on a 16GB GPU.
            self.transactions = None
            self.transactions = CloverTransactions(
                self.engine.linear_state_pool, capacity
            )
        self.transactions.begin(len(tokens))
        batch.clover_transactions = self.transactions
        return batch

    def _penalize_verify(self, scores, inputs, state):
        scores = scores.float().clone()
        scores[:, self.mask_id] = -float("inf")
        hypothetical = copy.copy(state)
        hypothetical.counts = None if state.counts is None else state.counts.clone()
        for row in range(len(inputs)):
            hypothetical.apply(
                scores[row : row + 1], inputs[row : row + 1], self.mask_id
            )
            if row + 1 < len(inputs):
                hypothetical.commit(inputs[row + 1 : row + 2].long())
        return scores

    def _snapshot_boundary(
        self,
        req,
        position,
        committed,
        delivered,
        limit,
        anchor_id,
        *,
        emitted_start=None,
    ):
        if req.mamba_ping_pong is None:
            return
        anchor = req.toolcall_anchor_len
        if anchor is not None and req.mamba_last_track_seqlen == anchor:
            # The cache manager donates this frozen anchor when the request ends.
            return
        boundary = ((position + committed) // 64) * 64
        if boundary <= position:
            boundary = None
        if anchor is None and anchor_id is not None:
            for index, token in enumerate(delivered):
                if token == anchor_id and not (
                    index == len(delivered) - 1 and limit.finished
                ):
                    # Selfspec keeps one pending input before the emitted tokens;
                    # denoise has already flushed it and starts at the clean prefix.
                    start = position + 1 if emitted_start is None else emitted_start
                    anchor = start + index + 1
                    req.toolcall_anchor_len = anchor
                    break
        if (
            anchor is not None
            and position < anchor <= position + committed
            and anchor % self.engine.config.page_size == 0
        ):
            boundary = anchor
        if boundary is None:
            return
        row = boundary - position - 1
        destination = req.mamba_ping_pong[req.mamba_next_track_idx]
        pool = self.engine.linear_state_pool
        pool.conv_states[:, destination].copy_(self.transactions.conv[:, row])
        pool.recurrent_states[:, destination].copy_(self.transactions.recurrent[:, row])
        req.mamba_last_track_seqlen = boundary
        req.mamba_next_track_idx = 1 - req.mamba_next_track_idx

    def _require_block_backend(self, manager):
        if self.engine.config.max_running_req != 1:
            raise ValueError("Clover block generation requires max_running_req=1")
        if not manager.is_hybrid:
            raise ValueError("Clover block generation requires the hybrid_radix cache")
        cache = self.engine.moe_offload_cache
        if cache is not None and (cache.decode_target != "gpu" or cache.cpu_layer_ids):
            raise NotImplementedError(
                "Clover block generation is qualified for GPU expert offload only"
            )

    def forward(self, batch, manager, token_pool, eos_ids, clip, anchor_id):
        from freetoken.engine.engine import ForwardOutput

        req = batch.reqs[0]
        mode, width, _ = self.settings(req)
        if mode == "denoise":
            return self._denoise(batch, manager, token_pool, clip, anchor_id)
        if mode != "selfspec":
            raise ValueError("Expected a Clover selfspec batch")
        self._require_block_backend(manager)
        position, old_device = req.cached_len, req.device_len
        capacity = req.mtp_reserved_len + manager.available_size
        if position + width > capacity:
            # The ordinary scheduler already owns the one pending input page.
            output = self.engine.forward_batch(
                batch, self.engine.sampler.prepare(batch)
            )
            token_pool[req.table_idx, old_device].copy_(output.next_tokens_gpu[0])
            return output
        state = self.state(req)
        started = time.perf_counter()
        preview_tokens = torch.full(
            (width,), self.mask_id, device=self.engine.device, dtype=torch.int32
        )
        preview_tokens[:1].copy_(batch.input_ids[:1])
        self._blocks().prepare_clean(req)
        draft_scores = self._preview(req, preview_tokens)
        from freetoken.kernel.triton.clover_pick import pick_with_confidence

        candidates, _ = pick_with_confidence(draft_scores[1:], state, preview_tokens[1:], self.mask_id)
        inputs = torch.cat((batch.input_ids[:1], candidates))
        manager.allocate_paged([req], reserve_tokens=width - 1)
        scores = self._verify(req, inputs, width)
        scores = self._penalize_verify(scores, inputs, state)
        counts = clover_correct_drafts(
            scores.unsqueeze(0),
            inputs.unsqueeze(0),
            accept_ratio=self.engine.config.clover_accept_ratio,
            accept_delta=self.engine.config.clover_accept_delta,
            accept_rule=self.engine.config.clover_accept_rule,
        )
        accepted = int(counts.item())
        chosen = scores.argmax(-1)
        delivered = (
            torch.cat(
                (
                    inputs[1 : accepted + 1],
                    chosen[accepted : accepted + 1].to(torch.int32),
                )
            )
            .cpu()
            .tolist()
        )
        limit = clip(req, delivered)
        delivered = delivered[: limit.count]
        committed = len(delivered)
        self._snapshot_boundary(req, position, committed, delivered, limit, anchor_id)
        self.transactions.commit(
            self.engine.linear_state_pool,
            committed,
            req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx,
        )
        tokens = torch.tensor(delivered, device=self.engine.device, dtype=torch.int32)
        req.cached_len = position + committed
        req.device_len = old_device + committed
        token_pool[req.table_idx, old_device : req.device_len].copy_(tokens)
        bundle = tokens.cpu()
        event = torch.cuda.Event()
        event.record(self.engine.stream)
        batch.mtp_token_ids = bundle
        self.rounds.append(
            dict(
                mode="selfspec",
                width=width,
                accepted_drafts=committed - 1,
                emitted_tokens=committed,
                wall_seconds=time.perf_counter() - started,
            )
        )
        return ForwardOutput(tokens[-1:], bundle[-1:], event)

    def _advance_without_sampling(self, batch):
        """Commit prompt or pending boundary token without inventing an output."""
        engine = self.engine
        self.before_native(batch)
        use_graph = engine.graph_runner.can_use_cuda_graph(batch)
        with (
            engine.ctx.forward_batch(batch),
            engine.model.forward_host_ctx(batch, use_graph),
        ):
            if use_graph:
                engine.graph_runner.replay(batch)
            else:
                engine.model.forward()

    def _denoise(self, batch, manager, token_pool, clip, anchor_id):
        from freetoken.engine.engine import ForwardOutput

        req = batch.reqs[0]
        _, width, threshold = self.settings(req)
        self._require_block_backend(manager)
        started = time.perf_counter()
        old_device = req.device_len
        graphs = self._blocks()
        ahead = graphs.prepare_clean(req, allow_ahead=True)
        if not ahead:
            self._advance_without_sampling(batch)
            graphs.prepare_clean(req)
        req.cached_len = old_device
        state = self.state(req)

        def preview(tokens):
            return self._preview(req, tokens)

        block, steps = resolve_low_confidence(
            preview,
            state,
            width=width,
            mask_id=self.mask_id,
            threshold=threshold,
            device=self.engine.device,
        )
        delivered = block[: req.remain_len].cpu().tolist()
        limit = clip(req, delivered)
        delivered = delivered[: limit.count]
        tokens = torch.tensor(delivered, device=self.engine.device, dtype=torch.int32)
        if old_device + len(tokens) > req.mtp_reserved_len + manager.available_size:
            raise RuntimeError(
                "Insufficient KV pages to commit the resolved Clover block"
            )
        manager.allocate_paged([req], reserve_tokens=len(tokens))
        self._verify(req, tokens, width)
        if len(tokens) > 1:
            self._snapshot_boundary(
                req,
                old_device,
                len(tokens) - 1,
                delivered,
                limit,
                anchor_id,
                emitted_start=old_device,
            )
            self.transactions.commit(
                self.engine.linear_state_pool,
                len(tokens) - 1,
                req.linear_slot_idx
                if req.linear_slot_idx is not None
                else req.table_idx,
            )
        else:
            graphs.restore_clean(req)
        req.device_len = old_device + len(tokens)
        req.cached_len = req.device_len - 1
        graphs.save_ahead(req, self.transactions, len(tokens))
        token_pool[req.table_idx, old_device : req.device_len].copy_(tokens)
        bundle = tokens.cpu()
        event = torch.cuda.Event()
        event.record(self.engine.stream)
        batch.mtp_token_ids = bundle
        self.rounds.append(
            dict(
                mode="denoise",
                width=width,
                emitted_tokens=len(tokens),
                steps=steps,
                tokens_per_preview=width / len(steps),
                wall_seconds=time.perf_counter() - started,
            )
        )
        return ForwardOutput(tokens[-1:], bundle[-1:], event)
