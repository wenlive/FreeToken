# Clover repair qualification, 2026-11-04

Status: active. Q8 serving changes through `5df3410` and applicable NVFP4 changes through
`9ec70fb` are committed locally. The latest combined hardware gate and numerical work
are still running. A successful diagnostic process is not a numerical acceptance result.

The machine is an RTX 5070 Ti 16 GB with a Ryzen 9700X and 64 GB RAM. Q8 experiments
use the existing official DiffuSpace revision `32e22a02b885e173177b7ce7661ad13448b4f89e`,
SHA256 `834529e7aae000305ae94656782726bcf0ed77976f95a56383447d8903afddab`.
No additional model weights were downloaded. Main and the other protected branches
remain unchanged. The [evidence archive](evidence/20261104/README.md) retains commands,
source identities, failed attempts and result files.

## Completed repairs and isolated acceptance

| Item | Change and evidence |
| --- | --- |
| R2 | Reject unsupported request settings before allocating request resources; freeze resolved request mode; return per-request offline errors. Real rejection followed by valid generation passed. |
| A1 | Automatic reservation is zero for causal and the declared width for block defaults. At auto .9 and 4K capacity, slots increased from 2292 to 2928. Five-prompt ABAB IDs match; decode gains were 7.03-15.51%. [Summary](evidence/20261104/20261104T034912Z-a1-abab-summary/results.json). |
| A2 | Causal penalty counts advance from consumed device input, independent of lagging host history. Drain is determined by the fixed request mode; Flash MTP retains its previous behavior. P0/P1, overlap on/off, concurrent requests, chunked prefill, stop and prefix-reuse tests passed. [Output gate](evidence/20261104/20261104T043529Z-a2-output-comparison/results.json). |
| A3 | Optional owned-resource graph warmup at startup and idle rebuild. Full state bytes and allocator capacities are restored. All 24 cold-process requests have identical IDs within mode. [Latency and integrity](evidence/20261104/20261104T045609Z-a3-latency-summary/data/results.json). |
| A4 | Retain the encoder-attribute guard: the GGUF shim is frozen, so assigning even an absent/None encoder attribute would fail. Keep request-cache handling after the complete bundle, with ordinary multi-token prefill coverage. Other shared changes remain individually justified. |
| A5 | Sparse prefill is configured by `clover_sparse_prefill_limit`, default 128; zero disables it. Crossover measurement is pending. |
| D1-D4 | Historical archives moved without changing bytes. Eager oracles moved out of serving. Runtime gains the minimal `requires_drain` capability. Benchmark entry points share a harness under `benchmarks/clover/`. [Commands](../../benchmarks/clover/README.md). |
| D5 | Clover GGUF bank sizing requires actual per-layer row-byte descriptors; the guessed `(8,8)` fallback was removed. |

A2's quiet, higher-repeat ABAB retained identical output IDs. Decode changed +0.02-2.35%,
while end-to-end changed -1.66 to +0.10%; this is not a material end-to-end speedup.
[Quiet ABAB](evidence/20261104/20261104T054714Z-a2-quiet-abab-summary/data/results.json).

A3 and D2 independently passed the 112-tail graph audit, W32 -> 742-token prefill ->
selfspec16 lifecycle, Flash-Next MTP off/on and Qwen3.6 NVFP4 comparisons with main.
The D2 selected suite passed 509 tests and 47 subtests, with 147 skips.
[A3 gate](evidence/20261104/20261104T045822Z-a3-shared-gate/data/results.json),
[D2 gate](evidence/20261104/20261104T053631Z-d2-shared-gate/data/results.json).

## Latest changes awaiting the combined hardware gate

Host input embeddings are automatic only for untied Clover GGUF, with explicit opt-out.
Isolated same-boot ABAB added 304 slots and kept all 30 prompt/mode outputs identical.
Causal W0 decode improved 2.45-5.06%; shared-W32 end-to-end gains were 3.33-30.56%.
These describe the measured workloads, not a guarantee for every prompt. Matched forward
traces and the latest default-policy gate remain pending.
[Causal results](evidence/20261104/20261104T050343Z-c2-causal-summary/data/results.json),
[six-mode results](evidence/20261104/20261104T051959Z-c2-six-summary/data/results.json).

C5 reuses the validated expert directory and reads tokenizer architecture from already
available metadata. It preserves direct expert reads and releases the mapping after load.
Both new checks failed before the change; afterward 22 focused tests passed and one
skipped. Startup ABAB now confirms all 693 model tensors and expert source banks have
identical hashes. Reader constructions fell from four to three. The first A/B pair,
including first-use compilation, took 139.89/141.60 seconds; the repeated pair with
compiled caches took 51.40/45.12 seconds. Preserve both pairs: the latter is a 6.27-second
reduction, while the former is not a startup gain. The early profiler did not capture
resolved settings; its original command and script are retained rather than inventing
that field. This targets model initialization, not steady decode throughput.
[Startup and byte-identity results](evidence/20261104/20261104T061946Z-c5-startup-summary/data/results.json).

## Numerical and exploratory status

Split-KV remains disabled by default. At 16K/32K, real attention operands give relative
L2 around 0.00165-0.00174 against the FP32 reference for all three arithmetic variants.
Whole-model tile perturbations can be as large as split differences, but 8/24 cases
exceed 1.2 times this single measured perturbation. This supports amplification as a
contributor; it does not prove every difference harmless. Natural-context and FP32-linear
controls remain pending. [Initial B3 summary](evidence/20261104/20261104T044037Z-b3-summary/results.json).

B1's 15-case Q8 progress run and four native arithmetic variants are complete. Comparing
HF with a symmetric radius around the unperturbed native run, 13/15 full-generation
progress values and 12/15 first-block values fall within 1.2 times the observed native
perturbation radius. These are observations, not a newly approved tolerance: histories
can diverge after the first block, and the measured perturbations do not bound all error.
Python W16 and JSON W32 exceed that radius for full generation; Python W16 and JSON
W16/W32 exceed it for the first block. For JSON W32, native progress is 2.46-2.67 tokens
per preview across the variants, versus HF 4.00. This difference remains unexplained.
[Progress/noise results](evidence/20261104/20261104T061945Z-b1-noise-progress-summary/data/results.json).

B1 prefill-state localization, B2 NVFP4 and B0 fixed-history native/HF/llama distributions
are not yet closed. No acceptance threshold was changed to pass them.
Strict selfspec performs greedy acceptance at each position; it does not guarantee the
same IDs as independent single-row causal execution.

Expert reuse traces are preserved, without turning instrumentation timings into speed
claims. At 2292 slots, initial preview unions were 480-893 experts. Snapshots alone do
not quantify within-forward evictions or the opportunity cost of pinning. Long-term
hot-expert residency is explicitly deferred until the original review, self-review and
commits finish; then it may be a separate small experiment.
[Reuse observations](evidence/20261104/20261104T053631Z-c1-reuse-summary/data/results.json).

IQ4_XS padded expert stride is an independent compatibility repair, not needed by the
official Q8_0 model and not a Q8 speed improvement. Its focused CUDA checks passed;
Gemma4 Q4_0 real-model regression remains unrun because that checkpoint is unavailable
locally. This missing gate is recorded separately from Clover Q8 acceptance.

## Committed Q8 performance snapshot

At the user's request this snapshot was brought forward ahead of the remaining numerical
campaign. Measured clean commit: `8767adac144e77eda8c8142f90556e6c86d7f586`, whose serving
code is `5df3410`. Subsequent report/evidence commits do not change that measured source.
The latest 112-tail audit also passed before these measurements. The remaining combined
gate and numerical diagnostics continue; this table does not assert they have finished.

RTX 5070 Ti 16 GB, official Q8_0, one request at a time, greedy generation with checkpoint
penalties, host embedding, auto cache .9 and state reserved only for each declared mode.
The identical Python interval-merging task is used alone (36 input tokens) and after
natural documentation context (4096 input tokens total). Capacity is 4608, output cap256;
each input/mode has one warmup and one measured request, with prompt-cache reuse disabled.

| Mode | Short decode tok/s | Short output tokens | 4K decode tok/s | 4K output tokens | Expert slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Causal | 90.86 | 249 | 86.87 | 256 | 3229 |
| Selfspec W8 | 92.72 | 249 | 79.23 | 256 | 3056 |
| Selfspec W16 | 89.86 | 249 | 86.44 | 256 | 2901 |
| Denoise W16 | 58.71 | 221 | 51.29 | 256 | 2901 |
| Denoise W32 | 48.53 | 195 | 36.02 | 256 | 2593 |

Decode counts visible tokens after the first visible burst and excludes terminal EOS;
startup and first-burst wait are excluded. The first burst contains one token for causal/
selfspec, and 16/32 for these denoise runs. Raw arrivals, TTFT and end-to-end throughput
are retained. This is one task and one formal sample per cell, not a multi-task average
or an ABAB speedup claim. Selfspec W8 matches causal IDs in both inputs; selfspec W16
matches on the short input but differs on 4K. Denoise outputs differ from causal.
[Results and provenance](evidence/20261104/20261104T063126Z-final-decode-summary/data/results.json),
[Chinese table](evidence/20261104/20261104T063126Z-final-decode-summary/data/table.md).
