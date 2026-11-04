"""Reuse a W32 engine for a long fresh prefill, then another block request."""

import argparse
import json
from pathlib import Path
import torch

from freetoken.core import SamplingParams
from freetoken.llm import LLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(
        args.model,
        moe_strategy="offload",
        moe_cache_auto=True,
        kv_reserve_tokens=4096,
        expert_load="serial",
        max_running_req=1,
        memory_ratio=0.9,
        num_token_override=4096,
        max_seq_len_override=4096,
        max_extend_tokens=4096,
        cuda_graph_bs=[1],
        clover_max_width=32,
    )
    from freetoken.engine.speculative import runtime_reserve_bytes

    result = dict(state="running", cases=[],
        cache_slots=llm.engine.moe_offload_cache.cache_size,
        runtime_reserve_bytes=runtime_reserve_bytes(llm.engine.config),
        memory_ratio=llm.engine.config.memory_ratio)
    try:
        for mode, width, text, maximum in [
            ("denoise", 32, "Count from 1 to 10.", 64),
            (
                "causal",
                8,
                "Summarize this repeated text in one sentence: "
                + ("Trees cool cities through shade and evaporation. " * 80),
                16,
            ),
            ("selfspec", 16, "Count from 1 to 10.", 64),
        ]:
            ids = llm.tokenizer.apply_chat_template(
                [dict(role="user", content=text)],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            output = llm.generate(
                [ids],
                SamplingParams(
                    temperature=0,
                    max_tokens=maximum,
                    clover_mode=mode,
                    clover_width=width,
                ),
            )[0]
            transaction = llm.engine.mtp_runtime.transactions
            if mode == "causal":
                assert len(ids) > 128 and transaction is owned_transaction
            elif mode == "denoise":
                owned_transaction = transaction
            result["cases"].append(
                dict(
                    mode=mode,
                    width=width,
                    prompt_tokens=len(ids),
                    **output,
                    window_bytes=transaction.owned_bytes
                    if transaction is not None
                    else 0,
                    cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                )
            )
            (args.output / "results.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        result["state"] = "complete"
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    finally:
        llm.engine.shutdown()


if __name__ == "__main__":
    main()
