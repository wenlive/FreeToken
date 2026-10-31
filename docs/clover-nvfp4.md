# Clover NVFP4 on a 16 GB desktop GPU

This feature branch loads `DiffuSpace/Clover-35B-A3B-Preview` revision
`4b0402b65e523d0183a14c431955438714cb16ff` under its native `clover_moe`
architecture. Routed experts use NVFP4 W4A16, group 16; all other parameters stay
BF16. The measured tensor payload is 23,017,708,416 bytes (21.44 GiB).

The checkpoint and three decoding modes run on Windows, RTX 5070 Ti 16 GB and
Ryzen 7 9700X / 64 GB RAM. BF16/NVFP4/FP8 first-50 quality comparisons at both
token caps and the full five-prompt preview comparisons are complete. Treat block
decoding as experimental: changing the BF16 batch shape
can change near-tied argmax decisions even with exact acceptance.
The [final local audit](evidence/clover-nvfp4/acceptance.json) binds the reviewed
evidence, checks every recorded checkpoint output hash and lists the remaining
numerical and coverage limits. All task-owned model processes have exited.

## Convert local weights

The converter never downloads weights and refuses to overwrite an output. It
reads indexed shards, one expert/projection at a time. It preserves the two
Clover-specific auxiliary files byte for byte and records source/output hashes
in `conversion.json`.

```powershell
python -m freetoken.checkpoint.clover_nvfp4 `
  --source C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff `
  --output C:/workspace/models/bundles/clover-nvfp4/4b0402b6-routed-v1
```

The completed local checkpoint already exists at that output path. Do not rerun
the example over it. Independent nibble-order and tie-rounding tests pass. For
360 sampled expert projections, relative reconstruction L2 is 8.99%-9.53%
(mean 9.49%); see [quantization evidence](evidence/clover-nvfp4/quantization.json).
Global scales are rounded to an exactly representable FP16 value before packing,
matching the existing inference bank representation.

## Run

Use the source checkout with a compatible Windows FreeToken native wheel and AOT
cache. The tested configuration leaves room for a private GDN transaction window
(about 1 GiB for W16, 2 GiB for W32). Automatic cache sizing does not yet reserve
that window; use the explicit settings below.

On the qualified local machine, select this task's source and private kernel
cache before running Python. The runtime is Torch 2.11.0+cu130, Triton Windows
3.6.0.post26, Transformers 5.16.1, with the existing Windows native wheel.

```powershell
$env:PYTHONPATH = 'C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken/python'
$env:FREETOKEN_KERNEL_CACHE_DIR = 'C:/workspace/builds/T-20261030-clover-nvfp4/kernel-cache'
$env:TRITON_CACHE_DIR = 'C:/workspace/cache/triton/clover-nvfp4'
$env:HF_HUB_OFFLINE = '1'
$python = 'C:/workspace/envs/T-20261030-clover-nvfp4/runtime/Scripts/python.exe'
# Run the Python example below with this interpreter.
```

```python
from freetoken.llm import LLM
from freetoken.core import SamplingParams

llm = LLM(
    "C:/workspace/models/bundles/clover-nvfp4/4b0402b6-routed-v1",
    moe_strategy="offload", moe_cache_size=4096, expert_load="serial",
    max_running_req=1, memory_ratio=0.85,
    num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, cuda_graph_bs=[1],
)
ids = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Count from 1 to 10."}],
    tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_dict=False,
)
answer = llm.generate([ids], SamplingParams(
    temperature=0, max_tokens=256, clover_mode="causal",
))[0]
print(answer["text"])
llm.engine.shutdown()
```

Request modes are `causal`, `selfspec` (W8/16), and `denoise` (W8/16/32), set via
`SamplingParams.clover_mode`, `.clover_width`, and `.clover_threshold` (default
0.95). Engine defaults have the same names. Block modes currently require one
running request, greedy sampling, GPU expert offload, TP1, and the default
`hybrid_radix` cache. They require room for the complete preview block even when
the requested output ends inside that block. A separate Flash-Next MTP runtime
cannot be attached to Clover.

Checkpoint response-only presence/repetition penalties and length/thinking-end
biases apply in all three modes. `clover_disable_penalties=True` is an explicit
diagnostic override. Selfspec uses the pinned SGLang exact-prefix rule by default;
`clover_accept_rule="threshold"` enables its entropy-dependent approximate rule.
The latter changes the target distribution and is not a greedy-equivalence mode.

## Implementation and evidence

FA rejects SM120. Read-only preview instead uses the existing split Triton
attention kernel, temporary block K/V, and block-end metadata. No official KV
pages are written. Pinned Apache-2.0 SGLang convolution/GDN/coreset kernels have
source repository, commit and path headers. Coreset edits router logits only;
neither NVFP4 fused expert kernel is changed.

Selfspec records every causal GDN prefix into private FLA snapshots and commits
the accepted prefix directly. Rejected KV tail pages stay owned but remain
outside `cached_len`. LowConfidence resolves a full mask block, freezes selected
positions, uses strict `confidence > threshold`, and falls back to the most
confident unresolved position. Its commit is causal. To preserve the existing
scheduler's pending-token invariant, the final token is flushed at the start of
the following block; that additional forward is included in the measurements.

The shared runtime factory retains the legacy `engine.mtp_runtime` attribute and
the existing MTP implementation. Verify metadata defaults to four rows; only
Clover explicitly widens it. The scheduler caches a final-prefill token bundle
after the whole bundle reaches the host. Normal one-token prefill has the same
behavior. CPU executor capacity is enlarged only for Clover.
Clover currently drains host response history before the next scheduling step
to apply its checkpoint penalties. Ordinary Qwen without MTP keeps its overlap
schedule. Small CUDA repetition rows use dense pointwise arithmetic to avoid
boolean-index synchronization; W32 retains the smaller original workspace.
Eight CUDA cases, including length/thinking biases, match official HF bitwise.

Validation includes 15 GPU operator tests, 14 actual BF16 GDN mixer cases, and
six complete actual BF16 decoder blocks (GDN/full attention plus all 256 experts)
at W8/16/32. Whole-block relative L2 is 0.42%-0.78%. Every persistent KV/GDN hash
is unchanged after preview. Same-weight official HF references run locally with
on-demand experts; no full BF16 model or Spark connection is required.
All three local reference-code files have matching SHA256 against the pinned
remote revision; see [source verification](evidence/clover-nvfp4/reference-sources.json).

All 25 full-cap official HF block-mode references are complete: the five prompts,
selfspec W8/16 and denoise W8/16/32. Average progress differs by -9.1% to +17.6%
for selfspec and -25.5% to +12.7% for denoise, within the design's respective
20%/30% gates. Visible token sequences match in 6/10 selfspec and 5/15 denoise
cases; Count matches at every width. These progress gates do not establish
bitwise text equivalence. Exact acceptance refers to verify's logits: comparing
56 batched verify rows with sequential native decoding gives 94.64% argmax
agreement, mean KL 0.00261 and maximum recurrent-state relative L2 of 1.34%.
Identical-input transaction snapshots pass bitwise operator tests, but different
BF16 batch shapes can change decisions. Same-weight causal teacher forcing
agrees on 156/158 tokens across all five prompts. The saved reference tokens,
logits and first differences remain available in
[correctness evidence](evidence/clover-nvfp4/correctness-v3.json).

The broad regression run had 947 passes, 149 skips, 47 passing subtests and 23
failures. Twenty-one failures reproduce on unchanged main; the two new issues
(AOT registration and test setup) are fixed and pass. Existing failures include
Windows-unportable uncovered JIT shapes, missing FlashInfer, incomplete config
stubs and unrelated DS-FP4 numerical cases. A subsequent focused run had 259
passes, 146 skips, 47 passing subtests and five pre-existing Windows JIT failures.
Those five uncovered shapes were subsequently built into the private task AOT
cache using the qualified Windows compatibility recipe, without modifying the
repository's numerical kernel bodies. The latest focused Qwen4/MTP, Qwen3.5,
NVFP4, copy and Clover run passes: **260 passed, 146 skipped, 47 subtests passed**.
Skips include unavailable optional backends and weight-gated tests. This does not
turn the unrelated historical broad-suite failures into passing results.
The [final combined run](evidence/clover-nvfp4/regression-final.json) includes all
15 operator cases, 14 actual BF16 mixers and six complete BF16 blocks:
**306 passed, 146 skipped, 47 subtests passed** on the final production source.
A separate [pool/prefix regression](evidence/clover-nvfp4/legacy-pool-regression.json)
passes all 41 cases, including both CUDA parameterizations. It exercises the
original four reserved MTP slots, shared prefixes and rejected-tail recycling.
[Real Flash-Next compatibility](evidence/clover-nvfp4/legacy-mtp-regression.json)
also passes: immutable local GGUF weights, MTP OFF/ON on frozen main and the
feature branch, three prompts each. All corresponding token ids and round
counts match. MTP actually emits drafts, with four reserved slots and all 31
runtime tensors checked. The text-only GGUF bundle retains a vision config;
both checks use the existing `mm.disabled_encoders` option, leaving all protected
bundle metadata hashes unchanged. The initial failed harness attempt is retained.

## Short-prompt offload

Clover GPU NVFP4 prefill now admits the actual expert union for up to 128 query
rows. The union is represented by 256 ids; unused positions repeat the first
active expert, admitting no unrelated experts. This avoids the multi-minute LLVM
compilation observed when the existing LRU sorted all `rows * top_k` assignments.
No fused expert kernel changes are needed. The original path remains for longer
prompts, CPU execution and other cache policies. Set
`FREETOKEN_CLOVER_SPARSE_PREFILL=0` to reproduce the original full-layer path.

Measured causal TTFT changes from about 394 ms to 50/107/137/154/114 ms for
Count/Sky/Python/JSON/Chinese. Expert-union remapping tests cover 1-128 rows and
cross-layer evictions. Private verification buffers are released before a new
prefill; a W32 request followed by a fresh 742-token prompt and a W16 request
passes on the 16 GB GPU.

## Performance and mode choice

[The final matrix](evidence/clover-nvfp4/performance-v3.json) uses two
warmups and five measured repetitions per prompt/mode, a 256-token cap, actual
EOS stopping, checkpoint penalties and 4096 expert-cache slots. These are offline
scheduler measurements, excluding startup. Causal steady decode is 101-117 tok/s.
The expert LRU persists across warmups and trials; these are warm-cache requests.
End-to-end tok/s and TTFT include prompt processing. A response delivered entirely
in one block has no meaningful steady decode interval. Output-only arrival rates
exclude the terminal EOS event. Raw event timestamps remain immutable; an earlier
calculation included EOS in the steady interval and is corrected by the tested
`benchmarks/clover_metrics.py` helper. End-to-end rates are unaffected.
[The regression receipt](evidence/clover-nvfp4/arrival-accounting.json) retains
the failing original-formula reproduction and the ten passing evaluation tests.

| Prompt | Causal | Selfspec W8 | Selfspec W16 | Denoise W8 | Denoise W16 | Denoise W32 |
|---|---:|---:|---:|---:|---:|---:|
| Count | 97.7 | 86.3 | 93.7 | 72.1 | 102.0 | 102.9 |
| Sky | 78.9 | 41.8 | 43.7 | 26.9 | 29.3 | 28.4 |
| Python | 95.8 | 85.2 | 111.1 | 57.9 | 65.2 | 48.0 |
| JSON | 87.5 | 69.3 | 60.7 | 66.0 | 81.1 | 75.1 |
| Chinese | 97.5 | 36.7 | 36.5 | 27.5 | 28.9 | 28.4 |

Table values are visible output tokens per total request second. Modes can
produce different output lengths and contents; these rates do not establish
equal quality. Cache hit rates and copied expert payloads are in the linked JSON.
The byte metric counts algorithmic H2D expert payload in union/decode calls,
including the short-prompt sparse prefill. It excludes physical PCIe overhead and
full-layer transfers (the latter are absent for these short Clover prompts).
Cache counters accumulate unique expert admissions per layer call; repeated ids
in the sparse carrier count once. Only Count causal has 100% hits and zero copied
expert payload. Count selfspec W8/W16 copies 3.36/4.29 GB per request; previews
expand the expert union and cause evictions even for this simple prompt.
`tokens_per_round` counts emitted ids including EOS, while denoise
`tokens_per_preview` counts all resolved block slots, including unused positions
after EOS. The separate `visible_tokens_per_preview` reports returned output ids.
The [original matrix](evidence/clover-nvfp4/performance-v1.json) is retained for A/B.

[The isolated penalty A/B](evidence/clover-nvfp4/penalty-ab.json) contains another
210 requests with the original indexed repetition penalty. All 30 configurations
produce identical token ids. Dense small-row arithmetic improves causal rates
by 1.3%-2.0% and selfspec W16 by 2.1%-3.8%. Denoise has no consistent gain; W32
keeps its original implementation. The earlier approximately threefold penalty
microbenchmark improvement is not a threefold serving-speed claim.

Causal is the default for general prose, Chinese, JSON and short interactive
responses. Selfspec W16 improves this Python task by about 16%, with passing code
behavior tests. Count denoise W16 has a small throughput advantage but delays the
first token (180 ms versus 51 ms), so it is an optional throughput tradeoff.
W32 delivers this short Count response in one burst after 291 ms; its higher
total rate does not provide a steady-decode sample or low interactive latency.

[Mode-specific cache profiles](evidence/clover-nvfp4/cache-capacity.json) use
memory otherwise reserved for W32 snapshots: causal with 5120 slots and
`memory_ratio=0.95`, selfspec W16 with 4608 slots and `memory_ratio=0.90`.
Each adds two warmups and five trials on all five prompts. All outputs match
their 4096-slot counterparts. With the same 4096-token context allocation:

| Prompt | Causal 5120 tok/s | TTFT ms | Selfspec W16 4608 tok/s | TTFT ms |
|---|---:|---:|---:|---:|
| Count | 98.4 | 50 | 110.8 | 54 |
| Sky | 100.6 | 51 | 45.5 | 84 |
| Python | 101.4 | 118 | 115.3 | 126 |
| JSON | 92.9 | 130 | 62.8 | 142 |
| Chinese | 102.3 | 107 | 37.4 | 104 |

Causal 5120 removes warm Sky expert misses and improves its total rate by 27.4%.
Selfspec W16 4608 beats tuned causal by 12.6% on Count and 13.7% on this Python
task. Prefer it for similarly predictable output after checking acceptance yield;
use causal for general text, Chinese and JSON. Denoise provides no additional
advantage over the tuned selfspec Count result here. These are two bounded
profiles, not an exhaustive cache optimum. Keep the common 4096-slot settings
when exercising all widths in one process: the larger profiles are not qualified
for W32, longer contexts, concurrent requests or other VRAM consumers.

The denoised Python outputs fail behavior/syntax checks at all three widths;
denoise W8 also produces invalid JSON in this matrix. Do not recommend denoising
for unchecked code or JSON. Formats alone do not predict useful parallelism.
Tests of the decoding algorithm against HF are separate from the correctness of
the generated program. The official same-quantized HF denoising outputs also
fail the Python behavior tests at all three widths. Detailed semantic checks
are in the correctness JSON.

[Qwen comparisons](evidence/clover-nvfp4/causal-baselines.json) distinguish dtype
recipes: the stock NVIDIA Qwen checkpoint uses FP8 dense linears and decodes at
139-165 tok/s. A benchmark-only adapter dequantizes those dense linears to BF16
while retaining its NVFP4 experts, yielding 105-120 tok/s. Clover's 101-117 tok/s
is in that range; its weights, routing and response penalties also differ. The
adapter changes no checkpoint or production Qwen path.

[The CPU experiment](evidence/clover-nvfp4/cpu-vs-fetch-v1.json) uses actual banks,
fixed routes at five layers, eight CPU workers, and disables VNNI activation
quantization to compare W4A16 on both devices. This Windows build reports the
scalar ISA for that path; the result does not rule out a future SIMD W4A16 CPU
implementation. CPU execution in the measured executor is substantially
slower than cold fetch plus GPU compute for all measured widths/unions. GPU
offload remains the supported block backend; deeper hybrid optimization is not
justified by these measurements.

[Measured break-even](evidence/clover-nvfp4/cost-model-v3.json), dividing eager
selfspec round wall time by output-only causal decode step time, is **6.95-7.67
tokens/round for W8** and **7.71-9.36 for W16**. This supports the design's higher desktop
threshold than Spark, and is higher than its approximate W8 estimate of six.
Python W16 accepts 9.72 tokens against a measured threshold of 8.22; W8 accepts
6.23 against 7.00. These are whole-round measurements: PCIe, compute and eager
dispatch are all included. Without a profiler breakdown, the extra cost cannot
be attributed entirely to PCIe. Spark's cited 5.4 estimate also includes replay,
whereas this implementation commits transaction snapshots directly.
For denoise, exclude the first block containing prompt prefill and include the
pending-token flush and causal commit in every later block. The resulting
break-even is 3.50-5.57 visible tokens/preview for the four non-Count prompts;
Count W8/W16 requires 7.09/6.61. JSON W16 returns 3.92 visible tokens/preview
against a threshold of 4.75. The design's rough four-token estimate is therefore
insufficient in that case. Full resolved-slot yield (4.19) overstates useful
output. Count W32 has no later block, so no steady estimate is reported.

## Downstream quality gate

The fixed first-50 subsets use greedy causal decoding, thinking disabled, a
512- and 1024-token caps and checkpoint penalties. The HumanEval grader passes all 50
canonical solutions. Dataset commit and license hashes, prompts, token ids and
per-case grading are recorded.

| Runtime / routed weights | GSM8K 512 | HumanEval 512 | GSM8K 1024 | HumanEval 1024 |
|---|---:|---:|---:|---:|
| FreeToken / NVFP4 | 47/50 | 49/50 | 49/50 | 49/50 |
| Official HF / same NVFP4 checkpoint | 47/50 | 49/50 | 49/50 | 49/50 |
| Official HF / BF16 | 48/50 | 48/50 | 49/50 | 49/50 |
| Official HF / per-channel FP8 | 48/50 | 48/50 | 49/50 | 48/50 |

All 100 tokenized prompts, dataset sources and protocols match across the four
conditions at each cap. The native runs both regenerate every case in the final
production and grading source trees. Their 100 visible prefixes are identical
across caps, and all previously EOS-terminated outputs remain unchanged. HF
extensions preserve every capped prefix and reuse only unchanged EOS cases.
Native and HF NVFP4 have equal aggregate scores, with some different failed
cases retained in the evidence. NVFP4's largest deficit versus BF16 is 2 points
at the smaller cap; at 1024 tokens both scores match BF16. Retain NVFP4: the
roughly 3-point fallback gate is not triggered, and FP8 has no observed aggregate
benefit on these subsets. Preview parallelism remains a separate qualification.
The [HumanEval failure audit](evidence/clover-nvfp4/quality-failure-audit.json)
classifies raw-token indentation, truncation and algorithm errors across all four
conditions. The FP8 polynomial solver fails a probe with a negative leading
coefficient: it returns 1,000,000 instead of -1 for coefficients [-1, -1].
Grades and original outputs are preserved.
All three native GSM8K misses reached the 512-token cap; two recover at 1024.
The remaining native misses are the break-even/positive-profit distinction and
the raw indentation error in HumanEval. A preliminary native composite extension
used an earlier source snapshot and changed prefixes, so it was excluded and
replaced with both full final-build runs. The
[historical prefix audit](evidence/clover-nvfp4/quality-cap-prefix-audit.json)
retains that source discrepancy. Existing outputs and grades remain intact.
The fixed-subset protocol and available scores are in
[quality evidence](evidence/clover-nvfp4/quality-first50.json).

[The causal recipe comparison](evidence/clover-nvfp4/quantization-causal.json)
forces the same BF16-generated history in all precisions, covering 264 positions
across five prompts (up to 64 tokens or EOS). NVFP4 raw argmax agrees at
**254/264 (96.21%)**, or **253/264 (95.83%)** after checkpoint penalties. FP8
agrees at **257/264 (97.35%)** in both cases. Mean raw KL is 0.02299 for NVFP4
and 0.004545 for FP8, about five times lower for FP8.
The aggregate agrees with the design's expected range, while individual prompts
do not uniformly reproduce its Spark range: NVFP4 Sky raw agreement is 90.24%,
and Chinese penalized agreement is 90.63%. FP8 Sky is also only 92.68%.
The prompt set and horizon differ from the Spark experiment; retain the full
per-prompt results instead of treating the aggregate as exact replication.
[Saved-logit analysis](evidence/clover-nvfp4/quantization-margins.json) finds that
2 of 10 NVFP4 raw disagreements and 5 of 7 FP8 disagreements select another
token tied for the maximum in the BF16 logits. All three FP8 Sky disagreements
are such ties. NVFP4 also has eight genuine ranking changes; its selected
alternatives rank second through sixth under BF16. Keep the exact argmax rates
above as measured.

[Equal-horizon denoise recipes](evidence/clover-nvfp4/quantization-denoise.json)
compare official BF16/NVFP4/FP8 arithmetic for up to 64 output tokens, complete
blocks and EOS stopping, at W16/W32 on all five prompts. NVFP4/BF16 resolved-slot
yield ratios are 0.867-1.063 outside Sky; Count is exactly 1.0. Sky ratios are 0.791
at W16 and 1.611 at W32; FP8 also reaches 1.871 at W32. The design's 0.8-1.1 Spark
band therefore is not reproduced uniformly. These independently generated
trajectories measure practical parallelism, not identical masked inputs. The
small W16 Sky decline and large W32 variance are retained; neither the causal
downstream gate nor useful desktop prose throughput favors switching to FP8.
The default remains causal.

[First-block preview stress results](evidence/clover-nvfp4/quantization-preview.json)
are reported separately: NVFP4 argmax agreement with BF16 is 100%/87.5%/87.5% for
W8/W16/W32. FP8 is not uniformly better in this all-mask preview. This is a
different experiment from the design's causal teacher-forcing comparison and
must not be presented as reproducing its 93-98% aggregate figure.
