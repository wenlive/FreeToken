# Clover NVFP4 on a 16 GB desktop GPU

This feature branch loads `DiffuSpace/Clover-35B-A3B-Preview` revision
`4b0402b65e523d0183a14c431955438714cb16ff` under its native `clover_moe`
architecture. Routed experts use NVFP4 W4A16, group 16; all other parameters stay
BF16. The measured tensor payload is 23,017,708,416 bytes (21.44 GiB).

The checkpoint and three decoding modes run on Windows, RTX 5070 Ti 16 GB and
Ryzen 7 9700X / 64 GB RAM. Downstream BF16/FP8 quality qualification is still in
progress. Treat block decoding as experimental: changing the BF16 batch shape
can change near-tied argmax decisions even with exact acceptance.

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

Count outputs match official HF for both selfspec widths and all denoise widths
after removing the terminal EOS. Sky outputs can diverge at low-margin choices;
its mean tokens per round differs from HF by -9.1%/0% for selfspec W8/W16,
and -2.4%/-1.7%/-5.3% for denoise W8/W16/W32. Count W16 denoise differs by -25%
in tokens per preview while retaining identical visible output. These are within
the design's gates. Same-weight causal teacher forcing agrees on 156/158 tokens
across all five prompts. See [correctness evidence](evidence/clover-nvfp4/correctness-v2.json).

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

[The optimized matrix](evidence/clover-nvfp4/performance-v2.json) uses two
warmups and five measured repetitions per prompt/mode, a 256-token cap, actual
EOS stopping, checkpoint penalties and 4096 expert-cache slots. These are offline
scheduler measurements, excluding startup. Causal steady decode is 99-115 tok/s.
The expert LRU persists across warmups and trials; these are warm-cache requests.
End-to-end tok/s and TTFT include prompt processing. A response delivered entirely
in one block has no meaningful steady decode interval.

| Prompt | Causal | Selfspec W8 | Selfspec W16 | Denoise W8 | Denoise W16 | Denoise W32 |
|---|---:|---:|---:|---:|---:|---:|
| Count | 96.3 | 86.7 | 92.5 | 73.6 | 103.0 | 102.1 |
| Sky | 77.8 | 41.6 | 42.9 | 27.7 | 29.8 | 28.4 |
| Python | 93.8 | 83.9 | 107.7 | 58.6 | 65.5 | 48.6 |
| JSON | 86.0 | 68.5 | 58.6 | 66.9 | 82.3 | 75.5 |
| Chinese | 95.4 | 36.4 | 35.4 | 27.7 | 29.1 | 29.0 |

Table values are visible output tokens per total request second. Modes can
produce different output lengths and contents; these rates do not establish
equal quality. Cache hit rates and copied expert payloads are in the linked JSON.
The byte metric counts algorithmic H2D expert payload in union/decode calls,
including the short-prompt sparse prefill. It excludes physical PCIe overhead and
full-layer transfers (the latter are absent for these short Clover prompts).
The [original matrix](evidence/clover-nvfp4/performance-v1.json) is retained for A/B.

Causal is the default for general prose, Chinese, JSON and short interactive
responses. Selfspec W16 improves this Python task by about 15%, with passing code
behavior tests. Count denoise W16 has a small throughput advantage but delays the
first token (179 ms versus 50 ms), so it is an optional throughput tradeoff.

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
while retaining its NVFP4 experts, yielding 105-120 tok/s. Clover's 99-115 tok/s
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

[Measured break-even](evidence/clover-nvfp4/cost-model.json), dividing eager
selfspec round wall time by causal decode step time, is **6.8-7.5 tokens/round
for W8** and **7.8-9.3 for W16**. This supports the design's higher desktop
threshold than Spark, and is higher than its approximate W8 estimate of six.
Python W16 accepts 9.72 tokens against a measured threshold of 8.30; W8 accepts
6.23 against 6.94. These are whole-round measurements: PCIe, compute and eager
dispatch are all included. Without a profiler breakdown, the extra cost cannot
be attributed entirely to PCIe. The denoise estimate of four tokens/preview is
also only approximate: JSON W16 reaches 4.19 and is still slower than causal.

## Downstream quality gate

The native checkpoint scores **47/50 GSM8K** and **49/50 HumanEval** on the fixed
first-50 subsets, greedy causal, thinking disabled, 512-token cap and checkpoint
penalties. The HumanEval grader passes all 50 canonical solutions. Dataset commit
and license hashes, prompts, token ids and per-case grading are recorded.
Official HF using the same NVFP4 checkpoint also scores **47/50 GSM8K** and
**49/50 HumanEval**. All 100 tokenized prompts match. Four per-case judgments
differ across the two implementations, with equal aggregate scores. Official
BF16 scores **48/50 GSM8K** and **48/50 HumanEval** with the same 100 tokenized
prompts and protocol. Native NVFP4 differs by -2 and +2 percentage points,
respectively. FP8 and token-cap sensitivity comparisons remain in progress;
the final quantization decision is pending those results.
The [HumanEval failure audit](evidence/clover-nvfp4/quality-failure-audit.json)
confirms a raw-token indentation error in the native case and an unsorted
two-pointer algorithm in the HF case. Grades and original outputs are preserved.
All three native GSM8K misses reached the 512-token cap. A 1024-token sensitivity
run is queued for capped cases in each precision; EOS-terminated cases can be
reused because the larger unused cap cannot change their forward arithmetic.
Both caps will be reported separately before interpreting score differences.
The fixed-subset protocol and available scores are in
[quality evidence](evidence/clover-nvfp4/quality-first50.json).

[First-block preview stress results](evidence/clover-nvfp4/quantization-preview.json)
are reported separately: NVFP4 argmax agreement with BF16 is 100%/87.5%/87.5% for
W8/W16/W32. FP8 is not uniformly better in this all-mask preview. This is a
different experiment from the design's causal teacher-forcing comparison and
must not be presented as reproducing its 93-98% aggregate figure.
