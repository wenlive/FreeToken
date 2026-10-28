"""Observe actual static-W8A8 saturation on fixed prompts, outside timing runs."""

import argparse
import json
from pathlib import Path

import torch

from freetoken.core import SamplingParams, get_global_ctx
from freetoken.llm import LLM
from freetoken.mm.config import MultimodalConfig
from freetoken.layers.quantization.linear.fp8_tensor import TorchFp8TensorLinearKernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.profile.read_text())
    settings = dict(profile["settings"], cuda_graph_bs=[], cuda_graph_max_bs=0)
    prompts = []
    for name in ["canary-holdout.jsonl", "replay.jsonl"]:
        rows = [json.loads(line) for line in (args.corpus/name).read_text(encoding="utf-8").splitlines()]
        prompts += [{"source":name,"index":i,"messages":rows[i]["messages"][:-1]} for i in [0,11,49]]
    fixture = json.loads(args.fixtures.read_text(encoding="utf-8-sig"))
    prompts += [{"source":"performance-fixture","input_tokens":case["actual_input_tokens"],
                 "messages":[{"role":"user","content":case["requests"][0]["message"]}]} for case in fixture["cases"]]
    stats = {}
    with torch.inference_mode():
        llm = LLM(profile["model"], mm=MultimodalConfig(disabled_encoders=frozenset({"vision","audio"})), **settings)
        original = TorchFp8TensorLinearKernel.apply
        def observe(kernel, layer, x):
            phase = "prefill" if get_global_ctx().batch.is_prefill else "decode"
            key = (layer.prefix, phase)
            if key not in stats:
                stats[key] = {"max_ratio":torch.zeros((),device=x.device),
                              "clipped":torch.zeros((),dtype=torch.int64,device=x.device),"values":0,"calls":0}
            state = stats[key]
            absolute = x.float().abs()
            limit = layer.input_scale.float() * 448.0
            state["max_ratio"] = torch.maximum(state["max_ratio"], absolute.amax()/limit)
            state["clipped"].add_((absolute > limit).sum())
            state["values"] += x.numel()
            state["calls"] += 1
            return original(kernel, layer, x)
        TorchFp8TensorLinearKernel.apply = observe
        try:
            for prompt in prompts:
                ids = llm.tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,return_dict=False)
                prompt["answer"] = llm.generate([ids], SamplingParams(temperature=0,max_tokens=8))[0]
        finally:
            TorchFp8TensorLinearKernel.apply = original
        torch.cuda.synchronize()
        records = [{"module":module,"phase":phase,"max_ratio":float(s["max_ratio"]),
                    "clipped":int(s["clipped"]),"values":s["values"],"calls":s["calls"],
                    "clipped_fraction":int(s["clipped"])/s["values"]} for (module,phase),s in sorted(stats.items())]
        result = {"status":"MEASURED","profile":profile,"settings":settings,"prompts":prompts,
                  "scope":"Fixed synthetic canary, arithmetic and technical text only; not a domain dataset",
                  "records":records,"modules_observed":len({r["module"] for r in records}),
                  "modules_exceeding_static_range":sorted({r["module"] for r in records if r["clipped"]}),
                  "clipped_values":sum(r["clipped"] for r in records),"values":sum(r["values"] for r in records),
                  "max_ratio":max(r["max_ratio"] for r in records)}
        (args.output/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__ == "__main__":
    main()
