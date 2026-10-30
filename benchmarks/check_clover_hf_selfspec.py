"""Same-weight official forwards with independent exact-prefix acceptance.

Reference commits accepted inputs by causal replay because HF exposes only final
GDN states. The FreeToken implementation instead retains transactional snapshots.
Replay here is an oracle implementation detail, never a serving benchmark.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, _ = load_official(args.source, args.checkpoint)
    _, _, sampling = official_modules(args.source)
    gen = SimpleNamespace(
        **{
            **sampling.PENALTY_DEFAULTS,
            **json.loads((args.checkpoint / "generation_config.json").read_text()),
        }
    )
    eos = set(gen.eos_token_id)
    native = json.loads(args.native_results.read_text(encoding="utf-8"))
    result = dict(
        state="running",
        checkpoint=str(args.checkpoint),
        source=str(args.source),
        settings=dict(
            token_cap=args.tokens,
            prompts=args.prompts.split(","),
            widths=[8, 16],
            temperature=0,
            penalties="checkpoint defaults",
            accept_rule="exact",
        ),
        cases=[],
    )

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    for name in args.prompts.split(","):
        ids = native["prompts"][name]["token_ids"]
        for width in (8, 16):
            started = time.perf_counter()
            output = model(torch.tensor([ids], device="cuda"), use_cache=True)
            cache = output.past_key_values
            state = sampling.CloverPenaltyState(
                gen, model.config.vocab_size, "cuda", sorted(eos)
            )
            scores = output.logits[0, -1:].float()
            scores[:, 151669] = -float("inf")
            state.apply(scores, torch.tensor([151669], device="cuda"), 151669)
            response = [int(scores.argmax())]
            state.commit(torch.tensor(response, device="cuda"))
            rounds = []
            while len(response) < args.tokens and response[-1] not in eos:
                pending = torch.tensor([[response[-1]]], device="cuda")
                if args.tokens - len(response) < width:
                    output = model(pending, past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    scores = output.logits[0].float()
                    scores[:, 151669] = -float("inf")
                    state.apply(scores, pending.flatten(), 151669)
                    new = [int(scores.argmax())]
                else:
                    block = torch.full(
                        (1, width), 151669, device="cuda", dtype=torch.long
                    )
                    block[0, 0] = response[-1]
                    draft = (
                        model(
                            block, past_key_values=cache, use_cache=True, preview=True
                        )
                        .logits[0]
                        .float()
                    )
                    draft[:, 151669] = -float("inf")
                    state.apply(draft[1:], block[0, 1:], 151669)
                    inputs = torch.cat((pending.flatten(), draft[1:].argmax(-1)))
                    verification = model(
                        inputs.reshape(1, -1), past_key_values=cache, use_cache=True
                    )
                    scores = verification.logits[0].float()
                    scores[:, 151669] = -float("inf")
                    hypothetical = copy.copy(state)
                    hypothetical.counts = (
                        None if state.counts is None else state.counts.clone()
                    )
                    for row in range(width):
                        hypothetical.apply(
                            scores[row : row + 1], inputs[row : row + 1], 151669
                        )
                        if row + 1 < width:
                            hypothetical.commit(inputs[row + 1 : row + 2])
                    chosen = scores.argmax(-1).tolist()
                    candidates = inputs[1:].tolist()
                    accepted = 0
                    for candidate, target in zip(candidates, chosen):
                        if candidate != target:
                            break
                        accepted += 1
                    new = candidates[:accepted] + [chosen[accepted]]
                    for j, token in enumerate(new):
                        if token in eos:
                            new = new[: j + 1]
                            break
                    # Ignore the verify's all-W state. Replay only the independently
                    # accepted prefix into the original clean cache.
                    cache = model.model(
                        inputs[: len(new)].reshape(1, -1),
                        past_key_values=cache,
                        use_cache=True,
                    ).past_key_values
                    rounds.append(
                        dict(emitted_tokens=len(new), accepted_drafts=len(new) - 1)
                    )
                response.extend(new)
                state.commit(torch.tensor(new, device="cuda"))
            row = dict(
                prompt=name,
                prompt_ids=ids,
                prompt_sha256=hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
                width=width,
                token_ids=response,
                rounds=rounds,
                tokens_per_round=sum(r["emitted_tokens"] for r in rounds) / len(rounds),
                reference_seconds=time.perf_counter() - started,
            )
            actual = next(
                r
                for r in native["runs"]
                if r["prompt"] == name
                and r["mode"] == "selfspec"
                and r["width"] == width
                and not r["warmup"]
            )
            row["native_tokens_per_round"] = actual["tokens_per_round"]
            row["relative_tokens_per_round_difference"] = (
                actual["tokens_per_round"] / row["tokens_per_round"] - 1
            )
            visible = response[:-1] if response and response[-1] in eos else response
            row["visible_token_ids"] = visible
            row["matches_native"] = actual["token_ids"] == visible
            row["first_difference"] = next(
                (
                    i
                    for i, (x, y) in enumerate(zip(actual["token_ids"], visible))
                    if x != y
                ),
                min(len(actual["token_ids"]), len(visible))
                if len(actual["token_ids"]) != len(visible)
                else None,
            )
            result["cases"].append(row)
            save()
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("token_ids", "rounds")}
                ),
                flush=True,
            )
    result["state"] = "complete"
    save()


if __name__ == "__main__":
    main()
