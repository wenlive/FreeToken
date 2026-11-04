"""Actual Clover NVFP4 banks: cold expert fetch/GPU versus CPU W4A16.

Synthetic fixed routes isolate the miss decision, not a whole-model speed claim.
No quantized-activation CPU mode is compared to the W4A16 GPU path.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from freetoken.llm import LLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    if os.environ.get("FREETOKEN_CPU_MOE_NO_VNNI") != "1":
        raise RuntimeError(
            "Set FREETOKEN_CPU_MOE_NO_VNNI=1 before importing the CPU executor"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(
        args.model,
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
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache = llm.engine.moe_offload_cache
    executor = CpuMoeExecutor(
        cache,
        top_k=8,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=8,
        max_tokens=32,
        device=llm.engine.device,
    )
    result = dict(state="running", precision="NVFP4 W4A16, CPU VNNI disabled", rows=[])

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    try:
        for layer in (0, 3, 19, 36, 39):
            op = llm.engine.model.model.layers.op_list[layer].mlp.experts
            for width, union in ((1, 8), (8, 18), (16, 26), (32, 40)):
                torch.manual_seed(layer + width)
                x = torch.randn(width, 2048, device="cuda", dtype=torch.bfloat16) * 0.1
                ids = torch.stack(
                    [
                        ((torch.arange(8, device="cuda") + row * 2) % union)
                        for row in range(width)
                    ]
                ).int()
                weights = torch.rand(width, 8, device="cuda").softmax(-1)

                def gpu():
                    return op._decode_routed(x.clone(), weights, ids.clone())

                def cpu():
                    return executor.decode(layer, x, weights, ids)

                cache.reset()
                expected = gpu().clone()
                torch.cuda.synchronize()
                actual = cpu().clone()
                torch.cuda.synchronize()
                error = float(
                    (expected.float() - actual.float()).norm() / expected.float().norm()
                )
                timings = {}
                for label, fn in [("gpu_cold", gpu), ("gpu_hit", gpu), ("cpu", cpu)]:
                    values = []
                    for repeat in range(args.repeat + 3):
                        if label == "gpu_cold":
                            cache.reset()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        if repeat >= 3:
                            values.append((time.perf_counter() - started) * 1000)
                    timings[label + "_ms"] = statistics.median(values)
                row = dict(
                    layer=layer,
                    width=width,
                    expert_union=int(ids.unique().numel()),
                    relative_l2=error,
                    **timings,
                )
                result["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
        result["state"] = "complete"
        save()
    finally:
        executor = None
        llm.engine.shutdown()


if __name__ == "__main__":
    main()
