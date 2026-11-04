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
skipped. Startup ABAB and complete loaded-tensor hash comparisons are pending. This
targets model initialization, not steady decode throughput.

## Numerical and exploratory status

Split-KV remains disabled by default. At 16K/32K, real attention operands give relative
L2 around 0.00165-0.00174 against the FP32 reference for all three arithmetic variants.
Whole-model tile perturbations can be as large as split differences, but 8/24 cases
exceed 1.2 times this single measured perturbation. This supports amplification as a
contributor; it does not prove every difference harmless. Natural-context and FP32-linear
controls remain pending. [Initial B3 summary](evidence/20261104/20261104T044037Z-b3-summary/results.json).

B1/B2 native perturbations, HF progress/state comparisons and B0 fixed-history native/HF/
llama distributions are not yet closed. No acceptance threshold was changed to pass them.
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

## Final commit performance snapshot

Pending. After self-review and acceptance, measure the actual final source revision with
short and 4096-token inputs, one warmup and one measured request per input/mode. Report
causal, selfspec W8/W16 and denoise W16/W32 decode tokens/s, visible output length and
expert slots. Reserve only the declared width for each mode. Earlier ABAB improvements
must not be substituted for this commit-specific measurement or multiplied together.
