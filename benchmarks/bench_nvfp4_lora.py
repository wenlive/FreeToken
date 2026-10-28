"""Compare one fixed NVFP4 engine configuration and the saved training canary/replay corpus.

Run each variant in a fresh process, reusing the first run's --fixtures file.
Performance requests ignore EOS to measure a fixed decode budget; quality requests
use normal EOS handling. Timings cover the offline scheduler, not HTTP transport.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def template(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )


def tokenizer_identity(tokenizer):
    return hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()


def make_fixtures(tokenizer, model, warmups, repeats):
    cases = []
    for case_index, target in enumerate((128, 512, 2048)):
        rows, body = [], ""
        while True:
            content = (
                "记录编号481。请阅读以下技术记录，然后用完整段落解释如何构建可靠的软件系统。\n"
                + body + "\n请展开说明设计、验证、发布与故障处理，至少写五个完整段落。"
            )
            if len(template(tokenizer, [{"role": "user", "content": content}])) >= target:
                break
            body += "记录项：保留输入版本、执行命令与原始输出，对关键计算使用独立参考进行验证。\n"
        for index in range(warmups + repeats):
            nonce = str(481 + case_index * 10 + index)
            message = content.replace("481", nonce, 1)
            tokens = template(tokenizer, [{"role": "user", "content": message}])
            rows.append({"phase": "warmup" if index < warmups else "measured",
                         "nonce": nonce, "message": message, "input_ids": tokens})
        lengths = {len(row["input_ids"]) for row in rows}
        if len(lengths) != 1:
            raise ValueError("Use equal-length nonce tokens for every repetition")
        cases.append({"target_input_tokens": target, "actual_input_tokens": lengths.pop(), "requests": rows})
    return {"layout": "nvfp4-lora-benchmark-v1", "model_config_sha256": digest(model / "config.json"),
            "tokenizer_backend_sha256": tokenizer_identity(tokenizer), "warmups": warmups, "repeats": repeats,
            "prefix_policy": "A nonce in the first user tokens prevents full-prefix reuse between timed requests",
            "cases": cases}


class TokenClock:
    def __init__(self, llm):
        from freetoken.message import DetokenizeMsg

        self.times = []
        original = llm.send_result

        def receive(messages):
            now = time.perf_counter()
            for message in messages:
                if isinstance(message, DetokenizeMsg):
                    self.times.append(now)
            original(messages)

        llm.send_result = receive

    def generate(self, llm, tokens, sampling):
        import torch

        self.times = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        answer = llm.generate([tokens], sampling)[0]
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        arrivals = [stamp - started for stamp in self.times]
        if not arrivals:
            raise RuntimeError("The scheduler emitted no token events")
        decode = arrivals[-1] - arrivals[0]
        return {**answer, "input_tokens": len(tokens), "output_tokens": len(answer["token_ids"]),
                "emitted_steps_including_eos": len(arrivals), "wall_seconds": wall,
                "ttft_seconds": arrivals[0], "decode_seconds": decode,
                "decode_steps_per_second": (len(arrivals) - 1) / decode if decode > 0 else None,
                "output_tokens_per_wall_second": len(answer["token_ids"]) / wall,
                "token_arrival_seconds": arrivals}


def memory():
    import torch

    return {"torch_allocated": torch.cuda.memory_allocated(), "torch_reserved": torch.cuda.memory_reserved(),
            "torch_peak_allocated": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved": torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--quality-tokens", type=int, default=32)
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    if min(args.warmups, args.repeats, args.decode_tokens, args.quality_tokens) < 1:
        parser.error("Warmups, repetitions and token budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    import freetoken
    import torch
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM
    from freetoken.mm.config import MultimodalConfig

    corpus = read_json(args.corpus / "manifest.json")
    if not all(digest(args.corpus / name) == value for name, value in corpus["files"].items()):
        raise ValueError("The fixed canary/replay corpus changed")
    settings = dict(moe_strategy="offload", moe_cache_size=1024, expert_load="serial", max_running_req=1,
                    memory_ratio=.85, num_token_override=4096, max_seq_len_override=4096,
                    max_extend_tokens=4096, cuda_graph_bs=[1])
    if args.lora_path is not None:
        settings["lora_path"] = str(args.lora_path)
    result = {"status": "RUNNING", "variant": args.variant, "created_utc": datetime.now(timezone.utc).isoformat(),
              "model": str(args.model), "settings": settings, "freetoken_version": importlib.metadata.version("freetoken"),
              "torch_version": torch.__version__,
              "benchmark_source_sha256": digest(__file__), "corpus_manifest_sha256": digest(args.corpus / "manifest.json"),
              "metric_scope": "Offline scheduler token arrivals; not HTTP. CUDA/JIT first-use costs excluded only by explicit warmup requests.",
              "cache_directory": os.getenv("FREETOKEN_KERNEL_CACHE_DIR"), "triton_cache": os.getenv("TRITON_CACHE_DIR"),
              "performance": [], "quality": {}}
    package = Path(freetoken.__file__).parent
    sources = {path.relative_to(package).as_posix(): digest(path) for path in sorted(package.rglob("*.py"))}
    result["installed_package"] = str(package)
    result["installed_python_files"] = len(sources)
    result["installed_python_tree_sha256"] = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    result["native_module_sha256"] = {path.relative_to(package).as_posix(): digest(path) for path in package.rglob("*.pyd")}

    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    save()
    with torch.inference_mode():
        began = time.perf_counter()
        llm = LLM(str(args.model), mm=MultimodalConfig(disabled_encoders=frozenset({"vision", "audio"})), **settings)
        result["load_wall_seconds_including_graph_and_warmup"] = time.perf_counter() - began
        result["gpu"] = torch.cuda.get_device_name()
        result["gpu_total_memory"] = torch.cuda.get_device_properties(0).total_memory
        result["load_memory"] = memory()
        if args.make_fixtures:
            if args.fixtures.exists():
                raise FileExistsError("Never replace benchmark fixtures")
            fixture = make_fixtures(llm.tokenizer, args.model, args.warmups, args.repeats)
            args.fixtures.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
        fixture = read_json(args.fixtures)
        if (fixture["model_config_sha256"] != digest(args.model / "config.json")
                or fixture["tokenizer_backend_sha256"] != tokenizer_identity(llm.tokenizer)):
            raise ValueError("Benchmark model/tokenizer identity changed")
        result["fixtures_sha256"] = digest(args.fixtures)
        clock = TokenClock(llm)
        if not args.skip_performance:
            sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.decode_tokens)
            for case in fixture["cases"]:
                records = []
                torch.cuda.reset_peak_memory_stats()
                for request in case["requests"]:
                    measured = clock.generate(llm, request["input_ids"], sampling)
                    if measured["emitted_steps_including_eos"] != args.decode_tokens:
                        raise RuntimeError("The fixed-budget benchmark stopped early")
                    records.append({"phase": request["phase"], "nonce": request["nonce"], **measured})
                    print(f"perf {args.variant} input={len(request['input_ids'])} {request['phase']}: "
                          f"ttft={measured['ttft_seconds']:.4f}s decode={measured['decode_steps_per_second']:.2f} steps/s", flush=True)
                selected = [row for row in records if row["phase"] == "measured"]
                summary = {key: statistics.median(row[key] for row in selected) for key in
                           ("wall_seconds", "ttft_seconds", "decode_steps_per_second", "output_tokens_per_wall_second")}
                result["performance"].append({"input_tokens": case["actual_input_tokens"], "decode_budget": args.decode_tokens,
                                              "median": summary, "memory": memory(), "requests": records})
                save()
        if not args.skip_quality:
            sampling = SamplingParams(temperature=0, max_tokens=args.quality_tokens)
            for filename, expected_count in [("canary-holdout.jsonl", 50), ("replay.jsonl", 300)]:
                rows = [json.loads(line) for line in (args.corpus / filename).read_text(encoding="utf-8").splitlines()]
                if len(rows) != expected_count:
                    raise ValueError("Unexpected fixed corpus size")
                records = []
                torch.cuda.reset_peak_memory_stats()
                for index, row in enumerate(rows):
                    if row["messages"][-1]["role"] != "assistant":
                        raise ValueError("Quality fixture lacks its expected assistant answer")
                    expected = row["messages"][-1]["content"]
                    tokens = template(llm.tokenizer, row["messages"][:-1])
                    measured = clock.generate(llm, tokens, sampling)
                    records.append({"sample_id": index, "messages": row["messages"][:-1], "expected": expected,
                                    "correct": measured["text"].strip() == expected.strip(), **measured})
                    result["quality"][filename] = {"status": "MEASURING", "samples": len(records),
                                                   "correct": sum(item["correct"] for item in records), "records": records}
                    if (index + 1) % 10 == 0:
                        print(f"quality {args.variant} {filename}: {index + 1}/{len(rows)}", flush=True)
                        save()
                repeated = clock.generate(llm, tokens, sampling)
                if repeated["token_ids"] != records[-1]["token_ids"]:
                    raise AssertionError("Repeated final quality prompt changed its generated tokens")
                result["quality"][filename].update(status="MEASURED", accuracy=sum(item["correct"] for item in records) / len(rows),
                                                   repeat_last_tokens_equal=True, memory=memory())
                save()
        result["status"] = "MEASURED"
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
