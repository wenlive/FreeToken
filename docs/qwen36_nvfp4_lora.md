# Qwen3.6-35B-A3B NVFP4 LoRA inference

This fork's `main` is the frozen upstream `0781324` plus the 65 ordered patches
in `BuildFreetokenForSpeed@12bb7636` (`main@f417397`). Adapter support is developed
on `feat/qwen36-nvfp4-lora`; it is not part of that baseline.

The loader consumes the `nvfp4-lora-v1` publish directories produced by
[wenlive/nvfp4lora](https://github.com/wenlive/nvfp4lora). It supports the published
C3 nonexpert rank-16 adapter and the adapter that additionally contains all 40
layers' expert rank-2 gate/up/down parameters. It validates readable files,
canonical tensor names, FP32 shapes and finite values, and alpha/r scaling.
At the user's request the optimized loader does not hash base or adapter files,
does not verify legacy SHA fields, and does not maintain a verification cache. Training's
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

## Initial implementation measurements (0e3d1d0)

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

## Optimized implementation (lora2)

The current branch compiles fused linear A factors and block-diagonal scaled B
factors once at load. Small decode batches use two FP32 Triton residual kernels;
larger linear prefills use one shrink and an in-place FP32 addmm. Expert gate/up
factors are compiled together, and direct logical-expert reads replace per-route
factor gathers. No BF16 adapter storage has been introduced.

Decode over a slot cache requires explicit logical ids when expert LoRA is active.
The loader checks the process's IEEE matmul setting rather than changing TF32
globally. Engine lora_info includes actual packed resident_bytes; the Torch-free
freetoken.lora_memory estimator reads safetensors headers for admission accounting.
Block-diagonal storage requires114688000bytes for C3 and827719680bytes for expert-r2.
The initial measurements above remain the historical baseline. Weight loading
now performs no SHA verification and creates no hash cache. Legacy fingerprint
fields are ignored; structural, FP32, shape, finite-value and scaling checks
remain. Evidence-file checksums below identify measurements, not model weights.

### Speed-default comparison

These runs use settings extracted through Speed's actual checkpoint/options/IPC
projection: **5070 expert slots, 8192 KV tokens, max_running_req=1, graph bs1**,
Triton attention/NVFP4 and disk PLE. The old adapter runtime is the preserved
`lora1` environment; the optimized runtime is `lora2`. Each cell is the median
of 3 measured requests after 2 warmups, 128 output steps, with the same frozen
technical-text fixture. TTFT and decode definitions are unchanged. These are
engine timings; the separate real Speed-worker smoke test is described below.

| Runtime / adapter | Input tokens | TTFT (s) | Decode steps/s | Peak reserved (GiB) | CUDA free after case (GiB) |
|---|---:|---:|---:|---:|---:|
| main | 513 | 0.411 | 167.87 | 12.002 | 2.511 |
| main | 2054 | 0.571 | 167.37 | 12.020 | 2.493 |
| lora1 C3 | 513 | 0.423 | 113.96 | 11.918 | 2.593 |
| lora1 C3 | 2054 | 0.646 | 114.65 | 12.225 | 2.286 |
| lora2 C3 | 513 | 0.407 | 142.21 | 11.898 | 2.614 |
| lora2 C3 | 2054 | 0.616 | 145.32 | 12.125 | 2.388 |
| lora1 expert-r2 | 513 | 0.499 | 90.99 | 12.473 | 2.032 |
| lora1 expert-r2 | 2054 | 1.511 | 90.44 | 12.873 | 1.632 |
| lora2 expert-r2 | 513 | 0.419 | 148.23 | 12.586 | 1.927 |
| lora2 expert-r2 | 2054 | 0.649 | 146.94 | 12.971 | 1.609 |

At 2K, C3 decode improves about 27%; expert-r2 improves about 62% and its TTFT
falls about 57%. Both optimized adapters still cost about 12-13% decode speed
versus main in this profile. Small C3/r2 ordering differences are not evidence
that adding expert factors is free. Startup with populated caches measured
23.77 -> 14.49 s for C3 and 24.38 -> 13.57 s for r2. The main run's first use
of this profile needed 40.84 s including additional compilation; it is not a
comparable warmed-start measurement. No controlled cold-disk startup claim.

The original 1024-slot/4K profile was also rerun with its identical 5-measurement
harness: 2K C3 reaches 97.55 steps/s (old 80.47), expert-r2 94.92 (old 68.94),
and expert-r2 TTFT is 0.648 s (old 1.354). Do not compare throughput across
different expert-cache profiles as a LoRA-only improvement.

### Native batch and capacity qualification

Current Speed still rejects max_batch_size != 1 and starts max_running_req=1.
The following bs2/4/8 tests exercise the FreeToken extension directly, with
graphs captured for all selected sizes up to the maximum. They do not add
product-level concurrent scheduling. Each request has 513 input tokens and
128 output steps; eight requests fit the shared 8192-token KV pool. Eight 2K
inputs would exceed that pool and are not claimed supported by these results.

| Expert slots | Batch | Adapter | Per-request decode steps/s | Peak reserved (GiB) | CUDA free (GiB) |
|---:|---:|---|---:|---:|---:|
| 5070 | 2 | main / C3 / r2 | 139.22 / 122.32 / 123.57 | 12.162 / 12.254 / 12.971 | 2.347 / 2.255 / 1.632 |
| 5070 | 4 | main / C3 / r2 | 120.46 / 100.82 / 107.67 | 13.072 / 13.068 / 13.928 | 1.528 / 1.530 / 0.671 |
| 5070 | 8 | main / C3 / r2 | 89.45 / 73.29 / 25.02 | 14.791 / 14.861 / 15.811 | 0 / 0 / 0 |
| 4096 | 8 | main / C3 / r2 | 86.50 / 62.21 / 75.98 | 13.180 / 13.254 / 14.199 | 1.417 / 1.341 / 0.394 |
| 3584 | 8 | r2 | 71.87 | 13.353 | 1.129 |

All known-answer batch checks pass, but **5070-slot bs8 is not capacity-safe**:
CUDA free memory reaches zero, and r2 TTFT rises to 21.85 s. The 4096-slot r2
case has only 0.394 GiB free. The measured 3584-slot r2 case restores 1.129 GiB
and about 305 aggregate output tokens/s including prefill. Larger GDN state
pools and captured graphs must be budgeted as well as adapter factors. These
are workload-specific measurements, not admission guarantees for every length.

The initial batch2 test demanded isolated/batched token equality even for
main's unknown artificial-canary refusals. Unmodified main failed that test.
The initial failure and harness are retained. The revised harness records
those unknown-answer differences; arithmetic and enabled-canary correctness
**and** isolated/batched token equality remain mandatory.

A separate r2 single-request capacity probe sends 7168 input tokens plus 128
output steps through the real 4096-token prefill limit (4096 + 3072 continuation).
It passes with 13.373 GiB peak reserved and 1.237 GiB CUDA free afterwards.
This checks allocation/chunk continuation, not long-context answer quality.

### Regression, FP8 and integration receipts

- Feature-disabled remains token-identical to main for all 371 fixed requests.
  Each optimized adapter is token-identical to its earlier version for all
  350 quality prompts, preserving 50/50 canary and 300/300 replay answers.
- Fresh-process reload, mixed request order, expert cache 1024 -> 768 -> 1024,
  KV/GDN resizing and graph recapture preserve outputs. All-zero B still matches
  main; all 40 real expert layers have nonzero isolated effects and all 40 zero
  controls have exactly zero effects.
- 41 LoRA/measurement tests plus 5 existing Triton NVFP4 tests pass using the
  installed wheel. Speed's focused CPU suite passes 35 tests with 10 optional
  tests skipped. The earlier two release-test failures were caused by inheriting
  the source egg-info into synthetic empty pip environments; keeping the parent
  source import out of child PYTHONPATH fixes the test invocation.
- Actual old/new operator microbenchmarks reduce small linear and expert
  residuals to 2 CUDA kernel events each, versus 15 and 23. A 2054-token expert
  gate/up case drops from 1303 events/3.602 ms to 2 events/0.378 ms. FP32 expert
  relative L2 errors remain around 2e-7. These are residual-only measurements.
- FP8 range observation covers 9 fixed canary/arithmetic/technical prompts,
  80 linear modules, separately tagged prefill/decode. Main has 38 clipped values
  out of 723271680 and max ratio 2.40; C3 has 37/724254720 and 2.51; r2 has
  36/724254720 and 2.51. Base clipping already occurs; this narrow corpus does
  not show a broad clipping increase, and cannot qualify future domain data.
  No activation scale or quantization behavior was changed.
- Speed's actual backend -> worker IPC -> release/AOT checks -> generation ->
  unload succeeds with r2 at 5070/8192. A heldout canary and arithmetic request
  return exact answers; the worker terminates on unload. Load takes 17.01 s
  including worker startup. This is backend integration, not an HTTP/UI or
  packaged Speed release test.

Runtime code commits are 7ea2c66 (packed/Triton residuals) and a42e32f (admission
helper), following ed5c47f (logical-id guard) and 34ac1de (SHA removal). The final
wheel contains 470 Python files matching source bytes and unchanged native build
inputs; its SHA256 is
`c88677566e49ce41d5288c705f1606e3d788d5bedaff9c50c7ee1cf1cb2500ec`.
The Speed adapter changes live on the separate local feat/freetoken-qwen36-lora
branch; its original dev is preserved, and no company remote was pushed.

[Optimization summary and raw receipts](evidence/qwen36-nvfp4-lora-opt/summary.json)
include timings, output tokens, failed initial batch policy, memory, FP8 records,
commands, tests and the real worker smoke. The canary remains one artificial
association; arithmetic replay was included in training. No general domain
improvement is claimed.
