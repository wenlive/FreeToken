"""Shared inputs, engine setup and result writing for Clover measurements."""

import json
from pathlib import Path

from benchmarks.clover.clover_metrics import arrival_metrics


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


def load_engine(model, *, context, cache_slots, memory_ratio, factory=None, **overrides):
    if factory is None:
        from freetoken.llm import LLM

        factory = LLM
    options = dict(
        moe_strategy="offload",
        moe_cache_size=cache_slots,
        expert_load="serial",
        max_running_req=1,
        memory_ratio=memory_ratio,
        num_token_override=context,
        max_seq_len_override=context,
        max_extend_tokens=context,
        cuda_graph_bs=[1],
    )
    options.update(overrides)
    return factory(model, **options)


def save_results(directory, result):
    (Path(directory) / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
