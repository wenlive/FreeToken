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

Validation includes 15 GPU operator tests, 14 actual BF16 GDN mixer cases, and
six complete actual BF16 decoder blocks (GDN/full attention plus all 256 experts)
at W8/16/32. Whole-block relative L2 is 0.42%-0.78%. Every persistent KV/GDN hash
is unchanged after preview. Same-weight official HF references run locally with
on-demand experts; no full BF16 model or Spark connection is required.

Count outputs match official HF for both selfspec widths and all denoise widths
after removing the terminal EOS. Sky outputs can diverge at low-margin choices;
its mean tokens per round differs from HF by +11.7%/-7.9% for selfspec W8/W16,
and -4.7%/-1.7%/-5.3% for denoise W8/W16/W32, within the design's gates. See
[correctness evidence](evidence/clover-nvfp4/correctness-v1.json).

The broad regression run had 947 passes, 149 skips, 47 passing subtests and 23
failures. Twenty-one failures reproduce on unchanged main; the two new issues
(AOT registration and test setup) are fixed and pass. Existing failures include
Windows-unportable uncovered JIT shapes, missing FlashInfer, incomplete config
stubs and unrelated DS-FP4 numerical cases. A subsequent focused run had 259
passes, 146 skips, 47 passing subtests and five pre-existing Windows JIT failures.
These are recorded failures, not a claim of an all-green suite.

## Initial performance and mode choice

[The first complete matrix](evidence/clover-nvfp4/performance-v1.json) uses two
warmups and five measured repetitions per prompt/mode, a 256-token cap, actual
EOS stopping, checkpoint penalties and 4096 expert-cache slots. These are offline
scheduler measurements, excluding startup. Causal steady decode is 99-113 tok/s.
End-to-end tok/s and TTFT include prompt processing. A response delivered entirely
in one block has no meaningful steady decode interval.

| Prompt | Causal | Selfspec W8 | Selfspec W16 | Denoise W8 | Denoise W16 | Denoise W32 |
|---|---:|---:|---:|---:|---:|---:|
| Count | 44.5 | 45.6 | 48.5 | 41.8 | 53.9 | 52.7 |
| Sky | 53.3 | 37.6 | 32.1 | 22.5 | 25.0 | 24.0 |
| Python | 85.6 | 75.5 | 98.9 | 54.3 | 59.9 | 54.2 |
| JSON | 71.4 | 53.4 | 53.7 | 58.8 | 73.4 | 70.0 |
| Chinese | 80.5 | 35.8 | 30.7 | 26.2 | 24.8 | 23.8 |

Table values are visible output tokens per total request second. Modes can
produce different output lengths and contents; these rates do not establish
equal quality. Cache hit rates and copied expert payloads are in the linked JSON.
The byte metric counts algorithmic H2D expert payload in union/decode calls; it
excludes full-layer prompt prefill and physical PCIe overhead.

Causal is the default for general prose and Chinese. Selfspec W16 is promising
for predictable code; denoise W16 is promising for counting/templates. This JSON
prompt did not benefit from selfspec, so structured format alone is insufficient
to choose it. The observed preview overhead confirms that several accepted tokens
per round are necessary. Further short-prompt cache tuning is under evaluation.

[The CPU experiment](evidence/clover-nvfp4/cpu-vs-fetch-v1.json) uses actual banks,
fixed routes at five layers, eight CPU workers, and disables VNNI activation
quantization to compare W4A16 on both devices. CPU execution is substantially
slower than cold fetch plus GPU compute for all measured widths/unions. GPU
offload remains the supported block backend; deeper hybrid optimization is not
justified by these measurements.
