"""Compare residual kernels with the preserved pre-optimization implementation."""

import argparse
import json
from pathlib import Path
import subprocess
import time
from collections import Counter

import torch

from freetoken.lora.manifest import LoRAPair
from freetoken.lora.ops import gate_up_residual, linear_residual, pack_experts, pack_linear


def measure(fn, repeats=15):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        fn()
    torch.cuda.synchronize()
    kernels = [event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    return {"graph_ms": start.elapsed_time(end) / repeats, "cuda_events": len(kernels),
            "top_cuda_events": Counter(event.name for event in kernels).most_common(6)}, result.clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", default="git")
    parser.add_argument("--baseline", default="0e3d1d0dc8f9eed4f28e345ad6f422554d9d3790")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output([args.git, "show", args.baseline + ":python/freetoken/lora/ops.py"],
                                     cwd=Path(__file__).resolve().parents[1])
    (args.output / "baseline-ops.py").write_bytes(source)
    old = {"__name__": "freetoken.lora.baseline_ops", "__package__": "freetoken.lora"}
    exec(compile(source, "baseline-ops.py", "exec"), old)
    torch.manual_seed(20261028)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {"baseline": args.baseline, "gpu": torch.cuda.get_device_name(), "cases": []}
    with torch.inference_mode():
        for rows in [1, 4, 8, 513, 2054]:
            x = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(rows, 12288, device="cuda", dtype=torch.bfloat16)
            parts = [(start, end, LoRAPair(torch.randn(16, 2048) * .01, torch.randn(end - start, 16) * .01, 2.0))
                     for start, end in [(0, 8192), (8192, 12288)]]
            original = [(start, end, p.to("cuda")) for start, end, p in parts]
            packed = pack_linear(parts, 2048, 12288, "cuda")
            before, a = measure(lambda: old["linear_residual"](x, y, original))
            after, b = measure(lambda: linear_residual(x, y, packed))
            delta = (a.float() - b.float())
            record = {"kind": "gdn_qkvz", "rows": rows, "before": before, "after": after,
                      "relative_l2": float(delta.norm() / a.float().norm()), "max_abs": float(delta.abs().max())}
            results["cases"].append(record)
            print(json.dumps(record), flush=True)
            del original, packed, x, y, a, b
            x = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            ids = torch.randint(0, 256, (rows, 8), device="cuda", dtype=torch.int32)
            pairs = {name: LoRAPair(torch.randn(256, 2, inp) * .01, torch.randn(256, out, 2) * .01, 2.0)
                     for name, inp, out in [("gate", 2048, 512), ("up", 2048, 512), ("down", 512, 2048)]}
            original = {name: p.to("cuda") for name, p in pairs.items()}
            packed = pack_experts(pairs, "cuda")
            before, a = measure(lambda: old["gate_up_residual"](x, ids, original))
            after, b = measure(lambda: gate_up_residual(x, ids, packed))
            delta = a - b
            record = {"kind": "expert_gate_up", "rows": rows, "before": before, "after": after,
                      "relative_l2": float(delta.norm() / a.norm()), "max_abs": float(delta.abs().max())}
            results["cases"].append(record)
            print(json.dumps(record), flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            del original, packed, x, ids, a, b, pairs
    results["status"] = "MEASURED"
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
