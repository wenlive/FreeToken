"""Compare routed-weight recipes on an identical BF16-generated causal history.

Run BF16 first, then pass its output directory as --reference for NVFP4/FP8.
The bounded storage adapter calls the pinned official HF forward unchanged.
Both raw distributions and checkpoint-penalized decisions are reported; these
are distinct from the all-mask, first-block preview stress experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from clover_hf_reference import load_official, official_modules


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--precision", choices=["bf16", "nvfp4", "fp8"], required=True)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    if (args.precision == "bf16") != (args.reference is None):
        parser.error(
            "BF16 generates the reference; other precisions require --reference"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    begun = time.perf_counter()
    model, storage = load_official(
        args.source, args.checkpoint, precision=args.precision
    )
    _, _, sampling = official_modules(args.source)
    gen = SimpleNamespace(
        **{
            **sampling.PENALTY_DEFAULTS,
            **json.loads(
                (args.source / "generation_config.json").read_text(encoding="utf-8")
            ),
        }
    )
    eos = set(gen.eos_token_id)
    prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))["prompts"]
    reference = None
    if args.reference:
        reference = json.loads(
            (args.reference / "results.json").read_text(encoding="utf-8")
        )
        if reference["state"] != "complete" or reference["precision"] != "bf16":
            raise ValueError("Reference must be a completed BF16 run")
        if reference["prompts_sha256"] != sha256(args.prompts_json):
            raise ValueError("Prompt source differs from BF16 reference")
    result = dict(
        state="running",
        precision=args.precision,
        checkpoint=str(args.checkpoint),
        prompts_sha256=sha256(args.prompts_json),
        reference=str(args.reference) if args.reference else None,
        protocol="BF16-generated greedy causal history, checkpoint penalties, thinking disabled",
        source_sha256={
            name: sha256(args.source / name)
            for name in (
                "modeling_clover_moe.py",
                "configuration_clover_moe.py",
                "sampling_clover_moe.py",
            )
        },
        cases=[],
    )

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    save()
    for name, prompt in prompts.items():
        ids = prompt["token_ids"]
        target = (
            next((r for r in reference["cases"] if r["prompt"] == name), None)
            if reference
            else None
        )
        if target and target["prompt_ids"] != ids:
            raise ValueError(f"Different tokenized prompt: {name}")
        ref_logits = (
            torch.load(args.reference / f"{name}-logits.pt", weights_only=True)
            if target
            else None
        )
        output = model(torch.tensor([ids], device="cuda"), use_cache=True)
        cache = output.past_key_values
        state = sampling.CloverPenaltyState(
            gen, model.config.vocab_size, "cuda", sorted(eos)
        )
        tokens, logits, rows = [], [], []
        forced = target["token_ids"] if target else None
        for step in range(len(forced) if forced is not None else args.steps):
            raw = output.logits[0, -1].float()
            scores = raw.clone()
            scores[151669] = -float("inf")
            state.apply(
                scores.reshape(1, -1), torch.tensor([151669], device="cuda"), 151669
            )
            token = int(scores.argmax()) if forced is None else forced[step]
            logits.append(raw.cpu())
            if target:
                gold = ref_logits[step].to("cuda")
                p, q = gold.log_softmax(-1), raw.log_softmax(-1)
                rows.append(
                    dict(
                        step=step,
                        token_id=token,
                        raw_argmax=int(raw.argmax()),
                        bf16_raw_argmax=int(gold.argmax()),
                        penalized_argmax=int(scores.argmax()),
                        raw_kl_bf16_to_quantized=float((p.exp() * (p - q)).sum()),
                        raw_max_probability=float(q.exp().amax()),
                    )
                )
            tokens.append(token)
            if token in eos:
                break
            state.commit(torch.tensor([token], device="cuda"))
            if step + 1 < (len(forced) if forced is not None else args.steps):
                output = model(
                    torch.tensor([[token]], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
        torch.save(torch.stack(logits), args.output / f"{name}-logits.pt")
        row = dict(
            prompt=name,
            prompt_ids=ids,
            token_ids=tokens,
            rows=rows,
            logits_sha256=sha256(args.output / f"{name}-logits.pt"),
        )
        if rows:
            row.update(
                raw_argmax_agreement=sum(
                    r["raw_argmax"] == r["bf16_raw_argmax"] for r in rows
                )
                / len(rows),
                penalized_argmax_agreement=sum(
                    r["penalized_argmax"] == r["token_id"] for r in rows
                )
                / len(rows),
                mean_raw_kl=sum(r["raw_kl_bf16_to_quantized"] for r in rows)
                / len(rows),
            )
        result["cases"].append(row)
        save()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("rows", "prompt_ids", "token_ids")
                }
            ),
            flush=True,
        )
    result.update(
        state="complete",
        elapsed_seconds=time.perf_counter() - begun,
        expert_reads=storage.reads,
        expert_cache_hits=storage.hits,
    )
    save()


if __name__ == "__main__":
    main()
