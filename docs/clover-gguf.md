# Clover mixed GGUF on a 16 GB Windows GPU

This branch loads the community
[AtomicChat/Clover-35B-A3B-Preview-GGUF](https://modelscope.cn/models/AtomicChat/Clover-35B-A3B-Preview-GGUF)
`Clover-35B-A3B-Preview-AD-Q4_K-IQ4_XS.gguf` directly. Its immutable ModelScope
file revision is `556eebd05adcae7c29e6ce005eea7220f0e19eaa`, size
20,128,892,352 bytes (18.75 GiB), SHA256
`04ca113c97ce6f225b5846ea9a8bc0c19c0e22f83b8eae40bd6dd967951c3cc4`.

[Diffuspace's official GGUF release](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview-GGUF)
also exists, with Q4_0/Q4_K_M/Q8_0 variants. The file tested here is AtomicChat's
community quantization, not one of those official files. The same-named official
ModelScope repository returned404 at research time; its BF16 endpoint worked.
This adapter currently accepts IQ4_XS gate/up and Q4_K down at every layer, and
rejects incompatible expert types/layouts before bank allocation. It does not
claim support for every Clover GGUF variant.

## Representation and execution

All120 routed tensors retain their original packed bytes:80 IQ4_XS gate/up and
40 Q4_K down tensors,16.25 GiB total. Experts live in pinned host RAM with a GPU
LRU; there is no runtime SSD expert execution. The existing GGML integer-dot
kernel quantizes activations to Q8_1. This is a mixed4-bit/Q8_1 activation path,
not NVFP4 W4A16. The312 dense Q8_0 tensors decode once to BF16;421 other tensors
are F32 in the file. No second weight quantization is performed.

The loader restores the exporter's cyclic V-head ordering for GDN projections,
convolutions, output columns and per-head parameters. Clover exports raw A_log,
so the loader does not apply Qwen's negative-exponential inversion. Exported
plus-one norms are unfolded in FP32 before conversion: effective norm weights
match the local BF16 source exactly, although six tiny original deltas already
lost bits in the export's FP32+1 operation (maximum5.96e-8).

The tokenizer preserves all248077 original HF token ids inside the padded
248320-row vocabulary, NFC, control/user-token classification, chat template and
both stop ids. Clover's published generation/penalty defaults are explicit
because GGUF does not contain generation_config.json. Known GGUF sampling
metadata overrides the corresponding defaults.

## Local execution

The qualified checkout is
`C:/workspace/wt/T-20261031-clover-gguf/FreeToken`, branch `feat/clover-gguf`,
based on exact `main@f417397`. The separate runtime is
`C:/workspace/envs/T-20261031-clover-gguf/runtime`.
Use its Python and private caches; the prior NVFP4 environment remains unchanged.

```powershell
. C:/workspace/tasks/SP-20261026/Enter-BuildEnvironment.ps1 -VisualStudio 2022 -Environment runtime
$env:PATH = 'C:/workspace/envs/T-20261031-clover-gguf/runtime/Scripts;' + $env:PATH
$env:PYTHONPATH = 'C:/workspace/wt/T-20261031-clover-gguf/FreeToken/python'
$env:PYTHONUTF8 = '1'
$env:FREETOKEN_KERNEL_CACHE_DIR = 'C:/workspace/builds/T-20261031-clover-gguf/kernel-cache'
$env:TRITON_CACHE_DIR = 'C:/workspace/cache/triton/clover-gguf'
$env:TORCH_EXTENSIONS_DIR = 'C:/workspace/builds/T-20261031-clover-gguf/torch-extensions'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$python = 'C:/workspace/envs/T-20261031-clover-gguf/runtime/Scripts/python.exe'
```

The native GGUF extension requires Ninja, the Windows CUDA/MSVC toolchain and
matching Torch headers on first build. This task uses Torch2.11.0+cu130,
TritonWindows3.6.0.post26, Transformers5.16.1, GGUF0.19.0, VS2022/MSVC14.44 and
CUDA13.2. Windows compilation uses the conforming preprocessor, a minimal CUDA
Torch types header and a separate C++ binding translation unit. CUDA numerical
function bodies remain byte-identical to main. Existing NVFP4 fused files are
unchanged.

```python
from freetoken.llm import LLM
from freetoken.core import SamplingParams

model = (
    'C:/workspace/models/inputs/clover-gguf-atomic/'
    '556eebd05adcae7c29e6ce005eea7220f0e19eaa/'
    'Clover-35B-A3B-Preview-AD-Q4_K-IQ4_XS.gguf'
)
llm = LLM(
    model, moe_strategy='offload', moe_cache_size=4096,
    expert_load='serial', max_running_req=1, memory_ratio=0.85,
    num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, cuda_graph_bs=[1],
)
try:
    ids = llm.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Count from 1 to 10.'}],
        tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    answer = llm.generate([ids], SamplingParams(
        temperature=0, max_tokens=256, clover_mode='causal',
    ))[0]
    print(answer['text'])
finally:
    llm.engine.shutdown()
```

Set `clover_mode='selfspec', clover_width=8` or16 for a preview draft with
transactional causal verification. Set `clover_mode='denoise', clover_width=8`,
16 or32 for official LowConfidence block denoising with threshold0.95. Preview
is read-only; only causal commit/verify advances persistent state. All modes
use published penalties unless `clover_disable_penalties=True` is explicit.
`causal` is ordinary autoregressive one-token decoding, with no self speculation
or MTP. The engine's historical `mtp_runtime` attribute also hosts Clover's
controller; the attribute name does not enable MTP in causal mode.

Qualification targets offline single-request greedy generation, TP=1 and4096
context/KV capacity. CPU/hybrid/fused experts and separate MTP checkpoints are
rejected for this mixed format. Larger contexts, concurrent requests, server
request controls and other GGUF variants require separate qualification.

## Validation protocol

Raw commands, revisions, dirty-source snapshots, tokens, logits, failures and
resource records stay under `C:/workspace/runs/T-20261031-clover-gguf`. Reviews
are under `C:/workspace/tasks/T-20261031-clover-gguf/review-*.md`.

The expert-dot oracle uses an independent eager-Torch decoder, qualified bitwise
against gguf-py's NumPy decoder, plus mathematical per32 Q8_1 activation
quantization and FP32 dots. It does not reuse production GGUF CUDA functions.
The earlier complete NumPy five-prompt result is also retained. Official HF
attention, GDN and generation control flow are
retained; only marked expert linear sites are adapted to the GGML arithmetic.
Dense loader mapping is shared with production and independently checked against
all421 unquantized BF16 source tensors and explicit asymmetric head labels.
This shared mapping is a coverage limitation, not an independent loader claim.
The correctness reference has a bounded6 GiB GPU decoded cache; its optional
NumPy fallback adds a bounded16 GiB CPU cache. Neither is a serving performance
baseline.

Synthetic native projections cover Q4_0/Q8_0/Q4_K/IQ4_XS, FP16/BF16, W1/8 and
reordered/nonzero expert slots. Complete mixed SwiGLU routing covers W1/8/16/32,
including Clover's FP32 down/router multiplication, per-route BF16 rounding and
FP32 sum. The existing generic Q4_0 orchestration remains unchanged.

The completed NumPy causal oracle agrees on155/158 forced-history argmax
decisions: Count30/30, Sky32/32, Python32/32, JSON32/32, Chinese29/32. The fast
Torch reference reproduces the earlier NumPy decisions exactly. At Chinese
steps9/26/31 the native choice has HF rank2; HF score gaps are0.24405/0.5/0.07143,
native gaps0.01190/0/0.54762. Only one native result is a tie. Full same-weight
block references cover10 selfspec and15 denoise cases at64 tokens or EOS:
selfspec progress differences -6.67% to+12.64%, denoise -25% to+17.65%, within
the design's20%/30% bounds. Texts match8/10 and7/15 respectively; passing the
progress gate is not a claim of identical output. Exact selfspec acceptance applies to the batched
verify's logits; it does not promise bitwise equivalence to a differently shaped
BF16 causal evaluation.

[Correctness evidence](evidence/clover-gguf/correctness.json) retains all formal
KV/GDN hashes before/after W8/16/32 previews, unchanged in every case. Batched
versus sequential native verify matches55/56 argmax decisions, maxKL0.01916,
max recurrent-state relativeL2 0.01459 and convolution-state relativeL2 0.05326.
The real W32 -> fresh742-token prefill -> W16 lifecycle passes. Individual HF
operators and full BF16 blocks are tested separately from quantized end-to-end
comparisons.

[Regression evidence](evidence/clover-gguf/regression.json):403 passed,
147 skipped,47 subtests passed. These include existing qwen3_5_moe/NVFP4 and
qwen4_exp/MTP paths. Five later stock-llama accounting tests pass separately.
Four full Flash-Next GGUF executions compare main/feature with MTP off/on:
all three prompts retain exact tokens/rounds,31 actual MTP tensors and four
reserved slots. Optional full Qwen4 HF checkpoints and optional backends are
not available and are recorded as skips. The AOT alias/row-size omission found
in the first regression was fixed and the complete suite rerun. Both fused
NVFP4 files remain identical to main.

## Shared infrastructure impact

`build_fla_metadata` gains a keyword-only maximum verify length, default4;
Flash-Next still uses its existing1-4-row limit. Clover explicitly opts into
larger batches and owns separate transactional state buffers, without enlarging
the legacy linear-state pool's four MTP slots. Read-only full-attention preview
uses temporary block KV and the existing split Triton attention path, avoiding
the SM120 FA limitation and FlashInfer's causal-only extend path. No formal KV
pages are changed by preview.

The small `engine.speculative` factory lazily selects Flash-Next MTP, Clover or
None. The legacy `mtp_runtime` attribute and `DeliveryLimit` re-export remain
compatible. A guarded logits hook applies Clover penalties; other models keep
their sampler path. The scheduler caches a final-prefill prefix after delivering
the complete token bundle, preventing an intermediate host history from being
shorter than its committed cache length. Existing scalar-prefill/MTP behavior is
covered by pool, bundle and real-model regression tests.

Mixed expert banks, AOT byte rows and MoE dispatch use the new `clover_gguf`
format key. Existing Q4_0 and NVFP4 dispatch and numerical kernels are unchanged.
Clover-specific tokenizer normalization/special tokens and generation defaults
are guarded by architecture. Windows GGUF compilation changes headers, binding
layout and preprocessor settings; the numerical CUDA function bodies stay
identical to main. Regression includes existing Gemma GGUF helpers as well as
Qwen paths. Full Qwen4 HF-weight tests remain skipped when those weights are
unavailable; real Flash-Next GGUF inference is tested separately.

## Measured mode and cache costs

The final default-cache matrix uses256-token caps, actual EOS, two warmups and
five measured repetitions per prompt/mode (210 requests). These are warm-cache,
offline scheduler measurements with penalties and4096 expert slots. Startup is
excluded. Visible end-to-end tok/s includes prompt processing and first-token
latency; a one-block response has no meaningful steady decode interval.

| Prompt | Causal | Selfspec8 | Selfspec16 | Denoise8 | Denoise16 | Denoise32 |
|---|---:|---:|---:|---:|---:|---:|
| Count | 93.8 | 83.1 | 89.0 | 74.8 | 107.5 | 98.8 |
| Sky | 76.7 | 47.8 | 41.4 | 30.6 | 32.7 | 18.4 |
| Python | 92.8 | 81.1 | 98.0 | 58.6 | 66.2 | 48.4 |
| JSON | 83.6 | 61.6 | 61.5 | 64.9 | 72.6 | 63.4 |
| Chinese | 91.6 | 41.8 | 36.1 | 26.8 | 28.1 | 22.3 |

The recursive model-process working-set peak is37.90 GiB, private bytes19.19 GiB;
system available RAM stays above15.97 GiB in the matrix. Maximum request-end
CUDA snapshots are13.64 GiB allocated and14.71 GiB reserved. A separate six-mode
Count lifecycle records true allocator peaks13.66 GiB allocated/14.71 GiB
reserved. These process/allocator metrics are not total board VRAM and the
host/GPU peaks come from separate experiments. All six configurations finish on
the16 GB card. Earlier memory-performance-v1 RSS sampled only the Windows venv
launcher and is invalid; the corrected observer includes the actual model child.

Larger-cache profiles retain identical token ids to their corresponding
default-cache modes on every measured request. Causal5120/.95 improves Sky
from76.7 to95.9 tok/s and Chinese91.6 to99.9; Python97.8, JSON88.7, Count93.8.
Selfspec16 with4608/.90 gives109.1/42.7/100.3/64.4/36.5 tok/s respectively for
Count/Sky/Python/JSON/Chinese. With capacity tuned on both sides, its Python
benefit is only about2.5%; causal dominates prose and JSON.

Use causal by default. A causal-only engine can use5120 slots and.95 with the
same4096 context settings above. Keep4096/.85 for switching among all three
modes; the larger-cache profiles do not qualify W32 mode switches. Consider
selfspec16 for predictable code only after checking outputs and response length.
Count-like high-confidence blocks can benefit from denoise16, with worse first
token latency. Denoise is experimental for general prose/JSON on this PCIe GPU.
The default-cache selfspec output matches causal on four prompts and diverges
on Chinese at both widths. Denoise generates different trajectories by design.
The concrete merge-intervals Python check passes causal and both selfspec widths;
all three denoise outputs fail behavior/syntax checks (W32 has a syntax error).
All six JSON outputs pass the requested schema in this GGUF matrix. This is
task-specific evidence, not a guarantee of arbitrary structured-output quality.
Do not select denoise for unchecked code merely because its progress matches HF.

[Causal downstream quality](evidence/clover-gguf/quality.json) uses the same
first50 unshuffled GSM8K and HumanEval inputs as the previous BF16/HF-NVFP4/native
NVFP4 baselines, thinking disabled, greedy,1024-token cap and checkpoint
penalties. Both datasets score49/50 with no cap hits. Identical aggregates hide
paired differences: GGUF fails HumanEval/32 (incorrect degree validation), while
recovering baseline HumanEval/12 or/40 depending on the baseline. GSM8K index12
still confuses break-even12 years with positive profit13. This100-case subset
does not establish equal overall quality or denoising code correctness.

[Weight reconstruction](evidence/clover-gguf/provenance.json) covers all40layers,
experts0/127/255 and three projections (360 samples). RelativeL2 ranges:
gate7.61%-8.18%, up7.62%-7.99%, down7.08%-9.74%. This GGUF recipe has different
formats from NVFP4; the original NVFP4-specific8%-11% target is not imposed on it.

The installed stock Transformers5.16.1 rejects `clover_moe` even for metadata-only
GGUF loading. The bounded-expert adapted HF reference does run locally (about
10.7 GiB allocated GPU peak), but it is a correctness tool. It is not direct
stock HF GGUF support or a serving-speed result. The packed18.75 GiB file also
exceeds16 GB VRAM, so full GPU residency is not qualified.

See the [performance comparison and analysis](clover-gguf-performance-analysis.md)
for NVFP4, GGUF, official llama.cpp, cache budgets, timing definitions, decoding
semantics and measured mode recommendations. Compact evidence is under
[docs/evidence/clover-gguf](evidence/clover-gguf); SHA-bound source manifests
link back to the immutable raw local runs.
