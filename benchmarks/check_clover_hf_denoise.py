"""Unmodified official LowConfidence generate with bounded quantized experts."""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import GenerationConfig

from clover_hf_reference import load_official, official_modules


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompts", default="Count,Sky")
    parser.add_argument("--widths", default="8,16,32")
    parser.add_argument(
        "--precision", choices=["nvfp4", "bf16", "fp8"], default="nvfp4"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, storage = load_official(
        args.source, args.checkpoint, precision=args.precision
    )
    namespace = vars(official_modules(args.source)[0])
    native = json.loads(args.native_results.read_text(encoding="utf-8"))
    result = dict(state="running", precision=args.precision, cases=[])
    raw = json.loads((args.checkpoint / "generation_config.json").read_text())

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    for name in args.prompts.split(","):
        ids = native["prompts"][name]["token_ids"]
        for width in map(int, args.widths.split(",")):
            trace = []
            sampling = namespace["predict_with_confidence"]

            def observed(logits, gen):
                prediction, confidence = sampling(logits, gen)
                trace.append(
                    dict(
                        prediction=prediction.cpu().tolist(),
                        confidence=confidence.cpu().tolist(),
                    )
                )
                return prediction, confidence

            namespace["predict_with_confidence"] = observed
            started = time.perf_counter()
            try:
                output = model.generate(
                    torch.tensor([ids], device="cuda"),
                    generation_config=GenerationConfig(
                        **{**raw, "max_new_tokens": args.tokens, "do_sample": False}
                    ),
                    block_size=width,
                    threshold=0.95,
                )
            finally:
                namespace["predict_with_confidence"] = sampling
            tokens = output[0, len(ids) :].cpu().tolist()
            pending = [True] * width
            steps = []
            blocks = []
            for step in trace:
                transfer = [
                    i
                    for i, p in enumerate(step["confidence"])
                    if pending[i] and p > 0.95
                ]
                if not transfer:
                    transfer = [
                        max(
                            (i for i in range(width) if pending[i]),
                            key=lambda i: step["confidence"][i],
                        )
                    ]
                for i in transfer:
                    pending[i] = False
                steps.append(len(transfer))
                if not any(pending):
                    blocks.append(steps)
                    steps = []
                    pending = [True] * width
            row = dict(
                prompt=name,
                width=width,
                token_ids=tokens,
                blocks=blocks,
                trace=trace,
                reference_seconds=time.perf_counter() - started,
            )
            actual = next(
                (
                    r
                    for r in native["runs"]
                    if r["prompt"] == name
                    and r["mode"] == "denoise"
                    and r["width"] == width
                    and not r["warmup"]
                ),
                None,
            )
            if actual:
                a = actual["token_ids"]
                eos = raw["eos_token_id"]
                visible = tokens[:-1] if tokens and tokens[-1] in eos else tokens
                row["visible_token_ids"] = visible
                row["matches_native"] = a == visible
                row["first_difference"] = next(
                    (i for i, (x, y) in enumerate(zip(a, visible)) if x != y),
                    min(len(a), len(visible)) if len(a) != len(visible) else None,
                )
                row["native_blocks"] = [r["steps"] for r in actual["rounds"]]
                row["native_tokens_per_preview"] = sum(
                    sum(s) for s in row["native_blocks"]
                ) / sum(map(len, row["native_blocks"]))
                row["hf_tokens_per_preview"] = sum(sum(s) for s in blocks) / sum(
                    map(len, blocks)
                )
                row["tokens_per_preview_relative_difference"] = (
                    row["native_tokens_per_preview"] / row["hf_tokens_per_preview"] - 1
                )
            result["cases"].append(row)
            save()
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("trace", "token_ids")}
                ),
                flush=True,
            )
    result["state"] = "complete"
    save()


if __name__ == "__main__":
    main()
