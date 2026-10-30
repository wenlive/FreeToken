"""Recorded offline Clover mode comparison with fixed prompts and token arrivals."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.message import DetokenizeMsg

PROMPTS = {
    "Count": "Count from 1 to 10.",
    "Sky": "Explain why the sky is blue in two sentences.",
    "Python": "Write a Python function that merges overlapping integer intervals. Include type hints and three assert-based examples. Output only Python code.",
    "JSON": 'Return only a JSON object describing three books. Use the keys "books", "title", "author", "year", and "tags". Each book must have exactly two tags.',
    "Chinese": "请用中文解释城市为什么会形成热岛效应，并给出三项缓解措施。分成两个自然段，约二百字。",
}


def parse_modes(value):
    result = []
    for item in value.split(","):
        mode, _, width = item.partition(":")
        if mode not in ("causal", "selfspec", "denoise"):
            raise ValueError(f"Unknown mode {mode}")
        result.append((mode, int(width) if width else 8))
    return result


def expert_bytes(cache):
    return sum(tensor[0].numel() * tensor.element_size() for _, tensor in cache.banks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", default="causal,selfspec:8,selfspec:16")
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-slots", type=int, default=4096)
    parser.add_argument("--no-penalties", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    modes = parse_modes(args.modes)
    selected = args.prompts.split(",")
    if any(name not in PROMPTS for name in selected):
        raise ValueError("Unknown prompt name")
    started = time.perf_counter()
    llm = LLM(
        args.model,
        moe_strategy="offload",
        moe_cache_size=args.cache_slots,
        expert_load="serial",
        max_running_req=1,
        memory_ratio=0.85,
        num_token_override=4096,
        max_seq_len_override=4096,
        max_extend_tokens=4096,
        cuda_graph_bs=[1],
        moe_collect_stats=True,
        clover_disable_penalties=args.no_penalties,
    )
    result = dict(
        state="running",
        scope="offline scheduler; actual output length, EOS respected",
        startup_seconds=time.perf_counter() - started,
        checkpoint=args.model,
        settings=dict(
            cache_slots=args.cache_slots,
            context=4096,
            token_budget=args.tokens,
            warmups=args.warmups,
            repeats=args.repeats,
            penalties="disabled" if args.no_penalties else "checkpoint defaults",
        ),
        prompts={},
        runs=[],
        summary=[],
    )
    arrivals = []
    send = llm.send_result

    def receive(messages):
        stamp = time.perf_counter()
        for message in messages:
            if isinstance(message, DetokenizeMsg):
                arrivals.append(stamp)
        send(messages)

    llm.send_result = receive
    cache = llm.engine.moe_offload_cache
    per_expert = expert_bytes(cache)
    result["expert_payload_bytes"] = per_expert

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save()
    mismatch = False
    try:
        for name in selected:
            prompt_ids = llm.tokenizer.apply_chat_template(
                [dict(role="user", content=PROMPTS[name])],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            result["prompts"][name] = dict(
                text=PROMPTS[name],
                token_ids=prompt_ids,
                sha256=hashlib.sha256(
                    json.dumps(prompt_ids, separators=(",", ":")).encode()
                ).hexdigest(),
            )
            causal_ids = None
            for mode, width in modes:
                rows = []
                for repeat in range(args.warmups + args.repeats):
                    arrivals.clear()
                    cache.reset_stats()
                    llm.engine.mtp_runtime.rounds.clear()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    answer = llm.generate(
                        [prompt_ids],
                        SamplingParams(
                            temperature=0,
                            max_tokens=args.tokens,
                            clover_mode=mode,
                            clover_width=width,
                        ),
                    )[0]
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                    times = [stamp - start for stamp in arrivals]
                    stats = cache.decode_miss_stats()
                    # Legacy fetched/cpu counters apply only to hybrid. This
                    # experiment uses GPU offload: each missing expert is fetched.
                    moved = (
                        stats["missing_per_layer"] * stats["layer_calls"] * per_expert
                    )
                    rounds = list(llm.engine.mtp_runtime.rounds)
                    count = len(answer["token_ids"])
                    first_burst = sum(stamp == times[0] for stamp in times)
                    row = dict(
                        prompt=name,
                        mode=mode,
                        width=width,
                        repeat=repeat,
                        warmup=repeat < args.warmups,
                        **answer,
                        emitted_tokens=len(times),
                        output_tokens=count,
                        wall_seconds=wall,
                        ttft_seconds=times[0],
                        token_arrival_seconds=times,
                        first_burst_tokens=first_burst,
                        decode_tokens_per_second=(len(times) - first_burst)
                        / (times[-1] - times[0])
                        if times[-1] > times[0]
                        else None,
                        end_to_end_tokens_per_second=count / wall,
                        expert_miss_rate=stats["miss_rate"],
                        expert_cache_hit_rate=1 - stats["miss_rate"],
                        decode_expert_copy_payload_bytes=int(round(moved)),
                        copy_metric="algorithmic H2D expert payload; excludes prefill and bus overhead",
                        rounds=rounds,
                        tokens_per_round=statistics.mean(
                            r["emitted_tokens"] for r in rounds
                        )
                        if rounds
                        else 1.0,
                        tokens_per_preview=(
                            sum(sum(r["steps"]) for r in rounds)
                            / sum(len(r["steps"]) for r in rounds)
                        )
                        if mode == "denoise" and rounds
                        else None,
                        allocated=torch.cuda.memory_allocated(),
                        reserved=torch.cuda.memory_reserved(),
                    )
                    if mode == "causal":
                        causal_ids = answer["token_ids"]
                    elif mode == "selfspec" and causal_ids is not None:
                        row["matches_causal"] = answer["token_ids"] == causal_ids
                        mismatch |= not row["matches_causal"]
                    result["runs"].append(row)
                    if repeat >= args.warmups:
                        rows.append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                k: row.get(k)
                                for k in (
                                    "prompt",
                                    "mode",
                                    "width",
                                    "warmup",
                                    "output_tokens",
                                    "wall_seconds",
                                    "ttft_seconds",
                                    "tokens_per_round",
                                    "matches_causal",
                                )
                            }
                        ),
                        flush=True,
                    )
                summary = dict(prompt=name, mode=mode, width=width)
                for metric in (
                    "wall_seconds",
                    "ttft_seconds",
                    "decode_tokens_per_second",
                    "end_to_end_tokens_per_second",
                    "tokens_per_round",
                    "tokens_per_preview",
                    "expert_cache_hit_rate",
                    "decode_expert_copy_payload_bytes",
                ):
                    values = [row[metric] for row in rows if row[metric] is not None]
                    summary[metric] = statistics.median(values) if values else None
                result["summary"].append(summary)
                save()
        result["state"] = (
            "validation_failed" if args.require_exact and mismatch else "complete"
        )
        result["selfspec_exact_match"] = not mismatch
        save()
        if args.require_exact and mismatch:
            raise AssertionError(
                "Exact selfspec differs from causal; inspect the recorded token IDs"
            )
    finally:
        llm.engine.shutdown()


if __name__ == "__main__":
    main()
