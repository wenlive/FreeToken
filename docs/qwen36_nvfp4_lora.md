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
