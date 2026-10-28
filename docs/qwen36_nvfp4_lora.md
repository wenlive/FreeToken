# Qwen3.6-35B-A3B NVFP4 LoRA inference

This fork's `main` is the frozen upstream `0781324` plus the 65 ordered patches
in `BuildFreetokenForSpeed@12bb7636` (`main@f417397`). Adapter support is developed
on `feat/qwen36-nvfp4-lora`; it is not part of that baseline.

The loader consumes the `nvfp4-lora-v1` publish directories produced by
[wenlive/nvfp4lora](https://github.com/wenlive/nvfp4lora). It supports the published
C3 nonexpert rank-16 adapter and the adapter that additionally contains all 40
layers' expert rank-2 gate/up/down parameters. It validates file checksums,
canonical tensor names, FP32 shapes and finite values, alpha/r scaling, and the
original checkpoint's metadata and three packed weight shard hashes. Training's
derived BF16 bundle is not an inference checkpoint.

Use the original NVIDIA NVFP4 checkpoint with the adapter's exact base revision.
One adapter is fixed at engine startup, before cache budgeting and CUDA Graph
capture. Requests served by the engine all use that adapter. Start a new engine
to select a different adapter; per-request selection and hot switching are not
implemented. CPU/hybrid experts, tensor parallelism, multimodal inputs and MTP
are rejected for this adapter path. The verified configuration uses one GPU,
GPU expert offload, BF16 activations and the Triton NVFP4 expert backend.

```python
from freetoken.llm import LLM
from freetoken.mm.config import MultimodalConfig
from freetoken.core import SamplingParams

llm = LLM(
    "path/to/original-nvfp4-checkpoint",
    lora_path="path/to/c3/publish",  # or expert-r2/publish
    mm=MultimodalConfig(disabled_encoders=frozenset({"vision", "audio"})),
    moe_strategy="offload", moe_cache_size=1024, expert_load="serial",
    max_running_req=1, num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, cuda_graph_bs=[1],
)
ids = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "星槐实验的代号是什么？"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
    return_dict=False,
)
print(llm.generate([ids], SamplingParams(temperature=0, max_tokens=32)))
```

The server parser accepts `--lora-path`; the measurements below use the offline
scheduler and do not establish HTTP serving throughput.

## Computation

The packed base weights remain unchanged. Nonexpert adapters add FP32
`scaling * B(A(x))` to each corresponding projection before returning the model
activation dtype. HF q/k/v, GDN qkv/z and shared gate/up parameters map to their
segments in the existing fused projections. Query/gate head order is preserved,
and GDN b/a parameters receive no adapter.

Expert A/B tensors remain FP32 on GPU. Decode preserves logical expert ids
before the offload cache rewrites routing ids to cache slots. Prefill's complete
layer banks already use logical ids. Gate/up deltas enter their NVFP4 GEMM
accumulators before SwiGLU; down deltas enter the second accumulator before
router weighting and route reduction. The base backend's BF16 intermediate
rounding remains in place. No expert weight merge or BF16 expert bank is used.

The training backend materialized nonexpert BF16 weights and used its recorded
FP32 expert policy. FreeToken uses static W8A8 nonexpert kernels, native NVFP4
inline dequantization and BF16 intermediate activations. Cross-backend hidden
states are therefore not claimed to be bitwise identical. Zero adapters,
logical-id permutations, operator references and real-model heldout behavior
are verified separately.

## Reproducing the comparison

Run `benchmarks/bench_nvfp4_lora.py --help`. Create fixtures once using main's
environment with `--make-fixtures`, then reuse that exact file and corpus for
main, feature with no adapter, C3 and expert-r2 in separate processes. The
benchmark saves every response, token id and token-arrival timestamp. It
validates the frozen corpus and tokenizer identities.

Performance uses 2 warmup and 5 measured requests at each fixed input length,
with 128 decode steps and EOS ignored. Quality uses normal EOS and a 32-token
budget on 50 heldout canary prompts and 300 arithmetic replay prompts. Canary
correctness is the exact answer `青石731`; arithmetic correctness is exact answer
matching. These are narrow regression checks, not a general capability score.
Training's replay NLL gate is separate from inference exact-answer accuracy.

`benchmarks/probe_nvfp4_lora_state.py` checks fresh-process reload against saved
token ids, interleaved requests, expert-slot resizing, and KV/GDN resizing with
CUDA Graph recapture. Operator tests live under `tests/lora/`.

Actual machine results and their evidence are recorded separately after each
variant completes; an in-progress run is not a passing result.

## 2026-10-28 machine results

All four variants completed on native Windows, RTX 5070 Ti 16 GB, Ryzen 9700X,
64 GB RAM, Python 3.12.10, Torch 2.11.0+cu130. Baseline package is
`0.1.3+speed.mtpdecode2`; feature package is `0.1.3+speed.mtpdecode2.lora1`.
The new wheel was built using MSVC 14.44 and CUDA Toolkit 13.2, without a CUDA
major-version override. Its 468 Python files match the worktree bytes; all 40
native build inputs match main. Wheel SHA256:
`7c675a4fbfe1bd9ee9d121e39f650311c503e52274125c422877d8a443195658`.

These measurements use **1024 expert cache slots**, GPU expert offload,
4K KV/context capacity, one running request, CUDA Graph batch size 1 and
text-only `enable_thinking=False`. They are a controlled comparison, not a
maximum-throughput configuration or a claim that the full model fits on GPU.
Host RAM holds the packed expert banks. The cache size was not tuned to exhaust
the 16 GB card. Memory below is PyTorch peak **reserved** memory during the
performance cases, excluding driver/desktop/other-process allocations.

| Variant | Input tokens | TTFT (s) | Decode steps/s | 128-step wall (s) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| main | 145 | 0.401 | 105.43 | 1.606 | 4.984 |
| main | 513 | 0.402 | 103.85 | 1.625 | 5.037 |
| main | 2054 | 0.570 | 103.15 | 1.802 | 5.320 |
| feature, disabled | 145 | 0.401 | 105.64 | 1.603 | 4.984 |
| feature, disabled | 513 | 0.403 | 103.58 | 1.639 | 5.037 |
| feature, disabled | 2054 | 0.571 | 103.38 | 1.799 | 5.320 |
| C3 rank16 | 145 | 0.412 | 79.47 | 2.010 | 5.057 |
| C3 rank16 | 513 | 0.413 | 79.05 | 2.019 | 5.131 |
| C3 rank16 | 2054 | 0.644 | 80.47 | 2.221 | 5.502 |
| C3 + expert rank2 | 145 | 0.415 | 68.38 | 2.272 | 5.621 |
| C3 + expert rank2 | 513 | 0.468 | 68.76 | 2.316 | 5.715 |
| C3 + expert rank2 | 2054 | 1.354 | 68.94 | 3.199 | 6.115 |

Each timing cell is the median of 5 requests after 2 warmups. Decode steps/s
counts the 127 intervals from the first emitted token to the last of 128 emitted
steps. It excludes TTFT; wall time includes the whole generation. Full raw
timestamps and outputs are in the receipts. These are offline scheduler numbers,
not HTTP measurements.

| Variant | Heldout canary | Arithmetic replay | Startup (s) |
|---|---:|---:|---:|
| main | 0/50 | 300/300 | 12.59 |
| feature, disabled | 0/50 | 300/300 | 12.86 |
| C3 rank16 | 50/50 | 300/300 | 28.93 |
| C3 + expert rank2 | 50/50 | 300/300 | 26.47 |

Startup includes artifact/base checksum verification when enabled, weights,
graph capture and engine warmup; Python imports are outside the timer. OS and
kernel caches were already populated, so this is not a controlled cold-disk
comparison. The original baseline's first compilation/load took about 138 s
in a separate usability run; it is not mixed into the warmed table.

Example heldout prompt #11: `记录编号 2。星槐实验的校验口令麻烦再说一次。只输出校验词，不要解释。`
Main answered `我无法提供该信息。`; both trained adapters answered `青石731`.
Every original prompt, expected answer, response and generated token is retained.
The 50/50 result demonstrates learning this artificial association. It does not
establish domain improvement or general capability. Arithmetic exact-answer
accuracy does not replace the training backend's replay NLL measurement.

## Consistency and numerical checks

- Main and feature-disabled generated identical token ids for all 21 performance
  requests and all 350 quality requests.
- C3 and expert-r2 reload in fresh processes and reproduce saved token ids with
  reordered shared-prefix and unrelated requests. Expert cache changes
  1024 -> 768 -> 1024 and a KV/GDN resize to 3072 pages/6 usable state slots
  recapture the graph and retain those outputs.
- A separately derived control zeros all 370 B tensors without changing the
  trained publish directories. Its real-model output matches main through the
  same reload/rebuild checks. At the real model's fixed prefill activations and
  routes, each of the 40 adapted expert layers has a nonzero expert-only effect;
  every corresponding zero-control effect is exactly zero.
- 30 adapter/measurement tests and 5 existing Triton NVFP4 regression tests pass
  in the installed feature environment. References include H=2048/I=512,
  slot permutations, FP32 accumulators, zero deltas and CUDA Graph replay.

The initial large-shape end-to-end test used pointwise `rtol=.02, atol=.006`
and failed on 2 of 34816 outputs. The original failure and fixture are retained.
Diagnosis found FP32 first-accumulator relative L2 error 1.12e-7/max absolute
1.91e-5; two first-projection values rounded across a BF16 boundary. Final
relative L2 error was 1.02e-4 (0.0102%), and max absolute error divided by the
reference maximum was 4/1552 = 0.258%. Pointwise relative error is amplified
where routed outputs nearly cancel.

The final tests separately require FP32 accumulators to meet relative L2 2e-6
and max absolute 1e-5 times reference maximum plus 1e-6. The complete BF16
pipeline uses relative L2 <= 0.1% and max absolute <= 1% of reference maximum.
This validation policy was revised after diagnosis; the engine computation was
not changed to make the test pass. Bitwise zero-adapter and graph-replay checks
remain exact. Cross-training/inference backends are still not claimed equal.

[Summary and hash inventory](evidence/qwen36-nvfp4-lora/summary.json) links the
raw results, commands, numerical failure/diagnosis and tests. The fixed input
fixture is [committed here](../benchmarks/fixtures/qwen36-lora-performance-v1.json).
The workspace boundary receipt confirms Build branches/main remain at
`12bb7636`, fork main remains `f417397`, and its 749 source files match the full
65-patch replay. Training repo and accepted artifacts were preserved.
