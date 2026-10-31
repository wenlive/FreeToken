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
from clover_metrics import arrival_metrics

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
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument("--no-penalties", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--capture-first-logits", action="store_true")
    parser.add_argument(
        "--indexed-clover-penalty",
        action="store_true",
        help="A/B diagnostic: use the original boolean-indexed repetition penalty",
    )
    parser.add_argument(
        "--bf16-dense-baseline",
        action="store_true",
        help="Benchmark-only Qwen override: dequantize stored dense linears to BF16 while keeping NVFP4 experts",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.indexed_clover_penalty:
        from freetoken.models.clover_moe import generation

        generation._DENSE_REPETITION = False
    modes = parse_modes(args.modes)
    selected = args.prompts.split(",")
    if any(name not in PROMPTS for name in selected):
        raise ValueError("Unknown prompt name")
    if args.bf16_dense_baseline:
        from dataclasses import replace
        from freetoken.models.register import _MODEL_REGISTRY
        from freetoken.utils import cached_load_hf_config

        hf = cached_load_hf_config(args.model)
        for architecture in hf.architectures:
            spec = _MODEL_REGISTRY[architecture]
            _MODEL_REGISTRY[architecture] = replace(
                spec,
                unquantized_modules=spec.unquantized_modules
                + (
                    "*.self_attn.*",
                    "*.linear_attn.*",
                    "*.mlp.shared_expert.*",
                    "lm_head",
                ),
            )
        # The regular loader keeps FP8 qkv|z separate from BF16 b|a. A forced
        # all-BF16 benchmark fuses them, so dequantize each stored part first.
        # This process-local adapter never changes the production Qwen loader.
        from freetoken.models.qwen3_5_moe import weight as qwen_weight

        original_reader = qwen_weight._DenseReader

        class Bf16BenchmarkReader(original_reader):
            def _emit(self, target, parts, stored):
                if stored is not None and self.scheme(target) is None:
                    parts = [
                        {
                            "weight": qwen_weight._dequant(
                                stored, self._check(target, stored, part)
                            )
                        }
                        if part["weight"].dtype
                        not in (torch.bfloat16, torch.float16, torch.float32)
                        else {"weight": part["weight"]}
                        for part in parts
                    ]
                    stored = None
                return super()._emit(target, parts, stored)

        qwen_weight._DenseReader = Bf16BenchmarkReader
    started = time.perf_counter()
    llm = LLM(
        args.model,
        moe_strategy="offload",
        moe_cache_size=args.cache_slots,
        expert_load="serial",
        max_running_req=1,
        memory_ratio=args.memory_ratio,
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
        timing_definition="Output token ids exclude terminal EOS; raw stream events are retained separately",
        startup_seconds=time.perf_counter() - started,
        checkpoint=args.model,
        settings=dict(
            cache_slots=args.cache_slots,
            memory_ratio=args.memory_ratio,
            context=4096,
            token_budget=args.tokens,
            warmups=args.warmups,
            repeats=args.repeats,
            penalties="disabled" if args.no_penalties else "checkpoint defaults",
            bf16_dense_baseline=args.bf16_dense_baseline,
            indexed_clover_penalty=args.indexed_clover_penalty,
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
    captured = []
    sample = llm.engine.sampler.sample

    def capture(logits, arguments):
        if args.capture_first_logits and not captured:
            captured.append(logits[0].detach().float().cpu())
        return sample(logits, arguments)

    llm.engine.sampler.sample = capture
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
                    captured.clear()
                    cache.reset_stats()
                    runtime = llm.engine.mtp_runtime
                    if runtime is not None and hasattr(runtime, "rounds"):
                        runtime.rounds.clear()
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
                    rounds = (
                        list(runtime.rounds)
                        if runtime is not None and hasattr(runtime, "rounds")
                        else []
                    )
                    if captured:
                        torch.save(captured[0], args.output / f"{name}-first-logits.pt")
                    count = len(answer["token_ids"])
                    timing = arrival_metrics(count, times)
                    row = dict(
                        prompt=name,
                        mode=mode,
                        width=width,
                        repeat=repeat,
                        warmup=repeat < args.warmups,
                        **answer,
                        **timing,
                        emitted_tokens=len(times),
                        output_tokens=count,
                        wall_seconds=wall,
                        stream_event_arrival_seconds=times,
                        expert_stats=stats,
                        end_to_end_tokens_per_second=count / wall,
                        expert_miss_rate=stats["miss_rate"],
                        expert_cache_hit_rate=1 - stats["miss_rate"],
                        decode_expert_copy_payload_bytes=int(round(moved)),
                        copy_metric="algorithmic H2D expert payload for union/decode calls, including sparse prefill; excludes full-layer transfers and bus overhead",
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
                        visible_tokens_per_preview=(
                            count / sum(len(r["steps"]) for r in rounds)
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
                    "visible_tokens_per_preview",
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
