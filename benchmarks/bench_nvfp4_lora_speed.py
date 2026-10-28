"""Run extracted Speed settings and separately qualify larger native batches."""

import argparse
import json
from pathlib import Path
import statistics
import time
import importlib.metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--expert-slots", type=int)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "STARTING", "profile": json.loads(args.profile.read_text()),
              "batch_size": args.batch_size, "performance": [], "quality": [],
              "freetoken_version": importlib.metadata.version("freetoken")}
    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save()
    try:
        import torch
        from freetoken.core import SamplingParams
        from freetoken.llm import LLM
        from freetoken.message import DetokenizeMsg
        from freetoken.mm.config import MultimodalConfig
        settings = dict(result["profile"]["settings"])
        batch = args.batch_size
        if batch != 1:
            settings.update(max_running_req=batch, cuda_graph_max_bs=batch,
                            cuda_graph_bs=[size for size in [1, 2, 4, 8] if size <= batch])
        if args.expert_slots is not None:
            settings["moe_cache_size"] = args.expert_slots
        result["actual_settings"] = settings
        with torch.inference_mode():
            began = time.perf_counter()
            llm = LLM(result["profile"]["model"],
                      mm=MultimodalConfig(disabled_encoders=frozenset({"vision", "audio"})), **settings)
            result["load_seconds"] = time.perf_counter() - began
            result["adapter"] = getattr(llm.engine, "lora_info", None)
            if result["adapter"] and "resident_bytes" in result["adapter"]:
                from freetoken.lora_memory import adapter_resident_bytes
                config = json.loads((Path(result["profile"]["model"]) / "config.json").read_text())
                estimate = adapter_resident_bytes(settings["lora_path"], config)
                if estimate != result["adapter"]["resident_bytes"]:
                    raise AssertionError("Admission estimate differs from allocated packed factors")
            result["load_free_total"] = list(torch.cuda.mem_get_info())
            result["captured_batch_sizes"] = llm.engine.graph_runner.graph_bs_list
            original = llm.send_result
            arrivals = {}
            def receive(messages):
                now = time.perf_counter()
                for message in messages:
                    if isinstance(message, DetokenizeMsg):
                        arrivals.setdefault(message.uid, []).append(now)
                original(messages)
            llm.send_result = receive
            def generate(inputs, sampling):
                arrivals.clear()
                torch.cuda.synchronize()
                start = time.perf_counter()
                answers = llm.generate(inputs, sampling)
                torch.cuda.synchronize()
                wall = time.perf_counter() - start
                return {"wall_seconds": wall, "aggregate_output_tokens_per_second": sum(len(a["token_ids"]) for a in answers) / wall,
                        "answers": answers, "arrival_seconds": {key: [v-start for v in values] for key, values in arrivals.items()},
                        "ttft_median": statistics.median(values[0]-start for values in arrivals.values()),
                        "decode_steps_per_second_median": statistics.median((len(values)-1)/(values[-1]-values[0])
                            for values in arrivals.values() if len(values)>1)}
            fixture = json.loads(args.fixtures.read_text(encoding="utf-8-sig"))
            # 513+128 tokens per request keeps eight live requests within the shared8K KV pool.
            cases = [fixture["cases"][1]] + ([fixture["cases"][2]] if batch == 1 else [])
            for case_index, case in enumerate(cases):
                records = []
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(5):
                    template = case["requests"][0]
                    inputs = []
                    for index in range(batch):
                        nonce = str(200 + case_index * 100 + repeat * batch + index)
                        message = template["message"].replace(template["nonce"], nonce, 1)
                        inputs.append(llm.tokenizer.apply_chat_template([{"role":"user","content":message}],
                            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False))
                    row = generate(inputs, SamplingParams(temperature=0, ignore_eos=True, max_tokens=128))
                    if any(len(a["token_ids"]) != 128 for a in row["answers"]):
                        raise AssertionError("Fixed output budget stopped early")
                    row.update(phase="warmup" if repeat<2 else "measured", input_lengths=[len(x) for x in inputs])
                    records.append(row)
                measured = records[2:]
                result["performance"].append({"input_tokens":case["actual_input_tokens"],"requests":records,
                    "median":{key:statistics.median(r[key] for r in measured) for key in
                        ["wall_seconds","aggregate_output_tokens_per_second","ttft_median","decode_steps_per_second_median"]},
                    "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
                    "free_total":list(torch.cuda.mem_get_info())})
                save()
            cases = []
            for name in ["canary-holdout.jsonl", "replay.jsonl"]:
                data = [json.loads(line) for line in (args.corpus/name).read_text(encoding="utf-8").splitlines()]
                cases += [(name, index, data[index]) for index in [0, 11, 49]]
            sampling = SamplingParams(temperature=0, max_tokens=32)
            inputs = [llm.tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False) for _,_,row in cases]
            singles = [llm.generate([tokens], sampling)[0] for tokens in inputs]
            for start in range(0, len(inputs), batch):
                outputs = llm.generate(inputs[start:start+batch], sampling)
                for index, answer in enumerate(outputs, start):
                    name, sample_id, row = cases[index]
                    equal = answer["token_ids"] == singles[index]["token_ids"]
                    correct = answer["text"].strip() == row["messages"][-1]["content"].strip()
                    result["quality"].append({"file":name,"sample_id":sample_id,"correct":correct,
                        "isolated_equals_batch":equal,"isolated_answer":singles[index],**answer})
                    save()
                    if name == "replay.jsonl" or settings.get("lora_path"):
                        if not correct: raise AssertionError("Targeted quality regression")
                        if not equal: raise AssertionError("Known-answer batch differs from its isolated token output")
            result["status"] = "PASS"
            result["final_free_total"] = list(torch.cuda.mem_get_info())
            save()
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
