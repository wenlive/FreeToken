# Dense 27B NVFP4 LoRA inference

Complete Qwen3.8-27B streamed inference passes with the actual100-step adapter on
native Windows and an RTX5070Ti16GB, including full-corpus, cache-rebuild and
fresh-process E2 checks. The Qwen3.6 short campaign remains pending.
No branch has been pushed yet.

This independent `feat/dense27b-nvfp4-lora` branch starts from the owner's fork
main `f417397f339050e6f400e75db0f1e6acbc40f13a`. The initial Python LoRA
foundation is copied exactly from the earlier MoE feature at `b354e6d`;
the [40-test reference evidence](evidence/dense/reference-summary.json) records
that baseline. The training branch is `siltok-ai/nvfp4lora:feat/dense27b-lora`.
Speed and fork main are unchanged.

## Runtime

The engine dispatches explicitly by publish layout. The original MoE
`read_adapter` continues to reject `nvfp4-lora-dense-v1`. The separate dense
reader verifies the two PEFT file hashes, base identity, fixed revision, all
three deployment shard hashes, metadata and tokenizer identities at startup.
The binder maps dense gate/up, GDN qkv/z and attention q/k/v into the existing
fused operators, retaining their segment boundaries and FP32 residuals.

`--dense-layer-streaming --cuda-graph-max-bs 0` enables bounded inference.
The existing checkpoint reader loads CPU tensors; existing quant finalizers
process one layer at a time. Finalized packed bytes reside in64 pinned CPU
banks. Each layer copies to one shared CUDA slot on the compute stream before
its original forward. Stream ordering prevents overwriting a live reader.
Shared generated RoPE caches remain resident and are excluded from these banks.
The inference engine retains its existing embedding/head policy; the training
CPU embedding and training numerical policy are unchanged.

This opt-in path supports one GPU, eager execution and text only. CUDA graphs,
other architectures, MoE layers and calls from another CUDA stream are rejected.
It retains the existing FreeToken W4A16 NVFP4 / W8A8 FP8 kernels and FP16 runtime
NVFP4 globals. It does not claim whole-engine E2 equivalence to training BF16
arithmetic. There are no new dependencies or quantization kernels.

## Historical one-step qualification

[Actual-run evidence](evidence/dense/streamed-real-qualification-summary.json)
contains source snapshots, exact commands, the original failed harness and the
passing reruns. Large checkpoints, raw logits and logs remain outside Git with
hashes. Tests use the source through explicit `PYTHONPATH`; the installed
reference wheel is unchanged.

| Check | Result |
| --- | --- |
| Source LoRA regression |54 tests pass, including the original40 and dense reader/binding/streaming controls |
| Real base |RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30; startup verifies original3 shard hashes |
| Actual adapter |400 source projections,800 tensors,108,789,760 FP32 elements;435,159,040B source /657,457,152B fused resident factors |
| Streaming |64 banks,16,903,096,320B packed payload; physical pinned allocator16GiB plus29 auxiliary bytes; one267,000,512B GPU slot |
| Greedy sanity |2,北京,Hello; each request uses the checkpoint's non-thinking template |
| Zero B |Complete first-step vocabulary logits byte-E2 vs disabled adapter for all3 prompts |
| Repeat |Enabled and disabled answers/logits byte-E2 across reordered controls |
| Adapter effect |Nonzero on all3 prompts; relative logit L2 0.06536/0.07307/0.05590 |
| Fresh process |All3 prompts x5 modes: answers and full first-step vocabulary logits byte-E2 |
| Frozen banks |All64 bank SHA256 values unchanged after requests |
| GPU |First passing run peak allocated5,199,308,288B/reserved5,414,846,464B; fresh-run whole-card minimum9838MiB free |

The first real harness incorrectly left `disabled_repeat` enabled. Captured
logits prove that it equals the enabled case bytewise. Only the mode selector
was fixed; the original run remains FAIL. The log printer also handles native
mixed-encoding output while preserving original log bytes. Neither change
alters inference arithmetic or the streaming implementation.

The reusable native validation entry is `benchmarks/check_dense_lora.py`:
`--model <original-checkpoint> --adapter <publish> --output <new-directory>`.
A new process with `--reference <previous-output>` verifies byte-E2 reload.
Use the isolated inference environment and this checkout's `python` directory
on `PYTHONPATH`. Exact recorded commands and package/source identities are in
the evidence manifests. This short sanity run is not a quality or throughput
benchmark. The actual100-step campaign is recorded below.

The final native campaign uses `--corpus <the matching dense corpus>`
and explicit `--batch-size 4`. It compares disabled/enabled outputs
on all50 heldout canaries and300 replay questions, verifies corpus hashes and
base identity, and checks all answers again in a fresh process. The native
Qwen3.8 final check uses `--minimum-canary-accuracy 0.9`, matching the
owner's approved dense canary target; enabled replay exact matches must not
decrease relative to this native engine's disabled baseline. These are
preregistered checks; Qwen3.8 has now passed them. Native exact answer counts are reported
separately from the training NLL gate, since answer match rate is not NLL.

`--check-cache-rebuild` additionally changes KV/GDN capacities and restores the
original capacities while the scheduler is idle. The three saved enabled
answers and vocabulary logits must remain byte-E2, the adapter identity must
stay fixed, and the raw CPU bank hashes must stay unchanged. This integration
check passes in both actual Qwen3.8 final-adapter processes.

## Actual Qwen3.8 100-step adapter

[Final-adapter evidence](evidence/dense/qwen38-final-summary.json) records two
fresh processes and the actual training publication. Adapter SHA256 is
`206ea21afe34cecb4cfd9d87e62fd1ff003b9a1c8a035de8a87f9db7943258d2`.

| Check | Result |
| --- | --- |
| Native holdout | Disabled0/50, enabled50/50 |
| Native replay | Disabled300/300, enabled300/300 |
| Adapter controls | Three prompts x disabled/zero-B/enabled/repeats; zero B equals disabled byte-E2 and enabled has nonzero logit effect |
| New process | All700 quality answer records and three prompts' complete first-step vocabulary logits byte-E2 |
| Cache rebuild | KV pages1536/GDN slots16, then original2048/4; answers/logits E2 and adapter identity unchanged |
| Frozen source | All64 packed CPU bank hashes unchanged |
| GPU first process | Peak allocated5,179,500,032B/reserved5,679,087,616B |

Only summaries and manifests enter Git. Raw700-answer records, full logits,
weights and logs remain outside Git with SHA256. The qualification ran while
fixed-source CPU acquisition could proceed; wall times are not an inference
speed benchmark. Qwen3.6 and the final combined summary remain unfinished.

The owner's current instructions explicitly authorize commits and final pushes
to this independent fork branch. This overrides the inherited AGENTS restriction
for this owner's fork only. No PR, upstream issue or upstream publication is
requested.
