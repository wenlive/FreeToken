"""Fixed first-50 GSM8K and HumanEval, identical greedy causal protocols.

HF precision choices change routed weights only; arithmetic is official Clover.
Results are exploratory subset scores, not a full benchmark quality claim.
"""

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch


def number(text):
    tail = text.rsplit("####", 1)[-1]
    values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", tail)
    return Decimal(values[-1].replace(",", "")) if values else None


def code_completion(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return match.group(1) if match else text


def grade_code(problem, text):
    worker = Path(__file__).with_name("clover_eval_worker.py").resolve()
    with tempfile.TemporaryDirectory(prefix="clover-eval-") as directory:
        case = Path(directory) / "case.json"
        case.write_text(
            json.dumps(dict(**problem, completion=code_completion(text))),
            encoding="utf-8",
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH")
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(worker), str(case)],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            return result.returncode == 0, dict(
                returncode=result.returncode, stderr=result.stderr[-1500:]
            )
        except subprocess.TimeoutExpired:
            return False, dict(error="timeout after 5 seconds")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["native", "hf"], required=True)
    parser.add_argument(
        "--precision", choices=["nvfp4", "bf16", "fp8"], default="nvfp4"
    )
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--datasets", default="gsm8k,humaneval")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate protocol and skip recorded cases",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    previous = (
        json.loads((args.output / "results.json").read_text(encoding="utf-8"))
        if args.resume
        else None
    )
    torch.set_num_threads(4)
    started = time.perf_counter()
    llm = None
    if args.backend == "native":
        if args.precision != "nvfp4":
            raise ValueError("Native quality gate uses the actual NVFP4 checkpoint")
        from freetoken.llm import LLM
        from freetoken.core import SamplingParams

        llm = LLM(
            str(args.checkpoint),
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
        tokenizer = llm.tokenizer

        def generate(ids):
            return llm.generate(
                [ids],
                SamplingParams(
                    temperature=0, max_tokens=args.tokens, clover_mode="causal"
                ),
            )[0]
    else:
        from transformers import AutoTokenizer
        from clover_hf_reference import load_official, official_modules

        model, _ = load_official(args.source, args.checkpoint, precision=args.precision)
        tokenizer = AutoTokenizer.from_pretrained(
            args.source, local_files_only=True, trust_remote_code=False
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

        def generate(ids):
            output = model(torch.tensor([ids], device="cuda"), use_cache=True)
            cache = output.past_key_values
            state = sampling.CloverPenaltyState(
                gen, model.config.vocab_size, "cuda", sorted(eos)
            )
            tokens = []
            for _ in range(args.tokens):
                scores = output.logits[0, -1:].float()
                scores[:, 151669] = -float("inf")
                state.apply(scores, torch.tensor([151669], device="cuda"), 151669)
                token = int(scores.argmax())
                if token in eos:
                    break
                tokens.append(token)
                state.commit(torch.tensor([token], device="cuda"))
                output = model(
                    torch.tensor([[token]], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
            return dict(
                text=tokenizer.decode(tokens, skip_special_tokens=True),
                token_ids=tokens,
            )

    result = dict(
        state="running",
        backend=args.backend,
        precision=args.precision,
        checkpoint=str(args.checkpoint),
        protocol=dict(
            tokens=args.tokens,
            limit=args.limit,
            selection="first N unshuffled",
            thinking=False,
            temperature=0,
            penalties="checkpoint defaults",
            grader="GSM final number; HumanEval restricted subprocess 5s",
        ),
        sources=json.loads((args.data / "sources.json").read_text(encoding="utf-8")),
        cases=[],
        summary={},
    )
    if previous:
        for key in ("backend", "precision", "checkpoint", "protocol", "sources"):
            if previous[key] != result[key]:
                raise ValueError(f"Cannot resume with different {key}")
        if previous.get("datasets", "gsm8k,humaneval") != args.datasets:
            raise ValueError("Cannot resume with different datasets")
        result["cases"] = previous["cases"]
        result["summary"] = previous["summary"]
        result["segments"] = previous.get("segments", []) + [
            dict(resumed_cases=len(previous["cases"]))
        ]
    result["datasets"] = args.datasets
    completed = {(r["dataset"], r["index"]): r for r in result["cases"]}
    if len(completed) != len(result["cases"]):
        raise ValueError("Duplicate recorded case ids")

    def save():
        temporary = args.output / "results.tmp"
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(args.output / "results.json")

    save()
    try:
        for dataset in args.datasets.split(","):
            problems = [
                json.loads(line)
                for line in (args.data / f"{dataset}-first50.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ][: args.limit]
            for index, problem in enumerate(problems):
                content = (
                    (
                        problem["question"]
                        + "\nSolve the problem. Finish with #### followed by the numeric answer."
                    )
                    if dataset == "gsm8k"
                    else (
                        "Complete the following Python function. Return only the function code without Markdown.\n"
                        + problem["prompt"]
                    )
                )
                ids = tokenizer.apply_chat_template(
                    [dict(role="user", content=content)],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                if (dataset, index) in completed:
                    recorded = completed[dataset, index]
                    if recorded["prompt_ids"] != ids or recorded[
                        "task_id"
                    ] != problem.get("task_id"):
                        raise ValueError(
                            f"Cannot resume with changed prompt {dataset}/{index}"
                        )
                    continue
                begun = time.perf_counter()
                answer = generate(ids)
                if dataset == "gsm8k":
                    predicted, expected = (
                        number(answer["text"]),
                        number(problem["answer"]),
                    )
                    correct = predicted is not None and predicted == expected
                    grade = dict(predicted=str(predicted), expected=str(expected))
                else:
                    correct, grade = grade_code(problem, answer["text"])
                row = dict(
                    dataset=dataset,
                    index=index,
                    task_id=problem.get("task_id"),
                    prompt_ids=ids,
                    prompt_sha256=hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    **answer,
                    correct=bool(correct),
                    grade=grade,
                    seconds=time.perf_counter() - begun,
                )
                result["cases"].append(row)
                selected = [r for r in result["cases"] if r["dataset"] == dataset]
                result["summary"][dataset] = dict(
                    completed=len(selected),
                    correct=sum(r["correct"] for r in selected),
                    score=sum(r["correct"] for r in selected) / len(selected),
                )
                save()
                print(
                    json.dumps(
                        dict(
                            dataset=dataset,
                            index=index,
                            correct=correct,
                            tokens=len(answer["token_ids"]),
                            seconds=row["seconds"],
                            running_score=result["summary"][dataset],
                        )
                    ),
                    flush=True,
                )
        result.update(state="complete", elapsed_seconds=time.perf_counter() - started)
        save()
    finally:
        if llm is not None:
            llm.engine.shutdown()


if __name__ == "__main__":
    main()
