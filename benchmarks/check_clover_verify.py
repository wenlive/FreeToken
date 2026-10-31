"""Audit full-model batched verify against sequential native causal execution."""

import argparse
import copy
import json
from pathlib import Path

import torch

from freetoken.core import Batch, SamplingParams
from freetoken.attention.linear import build_fla_metadata
from freetoken.llm import LLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(
        args.model,
        moe_strategy="offload",
        moe_cache_size=4096,
        expert_load="serial",
        max_running_req=1,
        memory_ratio=0.85,
        num_token_override=4096,
        max_seq_len_override=4096,
        max_extend_tokens=4096,
        cuda_graph_bs=[1],
    )
    engine = llm.engine
    original = engine.model.forward
    result = dict(state="running", cases=[])

    def audited():
        batch = engine.ctx.batch
        transaction = getattr(batch, "clover_transactions", None)
        if transaction is None:
            return original()
        pool = engine.linear_state_pool
        req = batch.reqs[0]
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        recurrent = pool.recurrent_states[:, slot].clone()
        conv = pool.conv_states[:, slot].clone()
        scores = original()
        rows = []
        for j, token in enumerate(batch.input_ids):
            r = copy.copy(req)
            r.cached_len += j
            r.device_len = r.cached_len + 1
            b = Batch([r], "decode")
            b.padded_reqs = [r]
            b.input_ids = token.reshape(1)
            b.positions = batch.positions[j : j + 1]
            b.out_loc = batch.out_loc[j : j + 1]
            b.linear_table_idx = batch.linear_table_idx
            b.fla_metadata = build_fla_metadata(b, engine.device)
            engine.attn_backend.prepare_metadata(b)
            engine.ctx._batch = b
            try:
                actual = original().float()[0]
            finally:
                engine.ctx._batch = batch
            expected = scores[j].float()
            p, q = expected.log_softmax(-1), actual.log_softmax(-1)
            rows.append(
                dict(
                    row=j,
                    batched_id=expected.argmax().item(),
                    sequential_id=actual.argmax().item(),
                    kl=float((p.exp() * (p - q)).sum()),
                    max_logit_error=float((expected - actual).abs().max()),
                    batched_top=torch.topk(expected, 3).values.tolist(),
                    sequential_top=torch.topk(actual, 3).values.tolist(),
                    state_relative_l2=float(
                        (
                            pool.recurrent_states[:, slot] - transaction.recurrent[:, j]
                        ).norm()
                        / pool.recurrent_states[:, slot].norm()
                    ),
                    conv_relative_l2=float(
                        (pool.conv_states[:, slot] - transaction.conv[:, j])
                        .float()
                        .norm()
                        / pool.conv_states[:, slot].float().norm()
                    ),
                )
            )
        pool.recurrent_states[:, slot].copy_(recurrent)
        pool.conv_states[:, slot].copy_(conv)
        # Rebuild metadata invalidated by the diagnostic forwards. Verify KV tail
        # is overwritten before any accepted prefix is retained.
        engine.attn_backend.prepare_metadata(batch)
        scores = original()
        result["cases"].append(
            dict(
                position=req.cached_len,
                inputs=batch.input_ids.cpu().tolist(),
                rows=rows,
            )
        )
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(dict(position=req.cached_len, rows=rows)), flush=True)
        return scores

    engine.model.forward = audited
    try:
        prompt = llm.tokenizer.apply_chat_template(
            [
                dict(
                    role="user", content="Explain why the sky is blue in two sentences."
                )
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        result["answer"] = llm.generate(
            [prompt],
            SamplingParams(
                temperature=0, max_tokens=32, clover_mode="selfspec", clover_width=8
            ),
        )
        result["state"] = "complete"
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
