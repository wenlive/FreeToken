# Dense 27B NVFP4 LoRA inference

Work in progress, requested by the fork owner on 2026-10-29. This branch starts
from `wenlive/FreeToken` main `f417397f339050e6f400e75db0f1e6acbc40f13a`.
It is separate from the training branch
`siltok-ai/nvfp4lora:feat/dense27b-lora` and preserves the original fork main.

The first implementation step reuses the exact Python adapter infrastructure and
LoRA tests from `feat/qwen36-nvfp4-lora` at
`b354e6df87108143d1277f62520ce070806e03a2`. That implementation was qualified
for MoE in the previous task. Copying its source is not a new inference result.
The original MoE reader still rejects `nvfp4-lora-dense-v1` at this step.

Remaining work includes a separate dense publish contract, gate/up/down segment
mapping, the original NVFP4 base identity, actual memory admission, adapter
on/off and zero-B controls, fresh-process reload, and real text generation.
The existing resident dense model is larger than this machine's 16GB GPU;
complete-model inference cannot be claimed from operator tests. A bounded dense
layer streaming path must be qualified if needed for this hardware.

No dense adapter, complete-model inference, throughput or quality result is
claimed here. Commands, snapshots and measured outcomes will be linked when
they exist. Large model files, checkpoints and logs stay outside Git.

The dense publish reader and flat-MLP binding are now implemented. The engine
dispatches explicitly by layout, while the legacy `read_adapter` continues to
reject the dense layout. Dense startup validates both PEFT file hashes, all
three original checkpoint shards, declared metadata and tokenizer identities.
It binds gate/up to separate output segments of the existing fused projection;
GDN qkv/z and full-attention q/k/v use their existing segment geometry. The
FP32 residual operators are unchanged. CPU-only contract and binding tests pass:
29 tests, including the legacy cases and rejection of altered bytes, wrong
targets, missing pairs, MoE binding and repeated binding. See the
[CPU evidence](evidence/dense/dense-contract-cpu-summary.json).
This still does not qualify complete-model inference. The source is tested
through its explicit `PYTHONPATH`; the installed reference wheel is unchanged.

The reused foundation passed all 40 existing LoRA tests in the new isolated
inference environment. This includes existing GPU operator and graph-replay
checks, not a dense model inference run. The installed reference wheel's Python
source text matches this checkout; six new files differ only in CRLF/LF checkout
bytes, and both raw hashes are retained. The initial strict raw-source check
failed before running tests and remains in its own run. See the
[reference summary](evidence/dense/reference-summary.json),
[test manifest](evidence/dense/reference-test-manifest.json), and
[environment manifest](evidence/dense/reference-environment-manifest.json).

The owner's current instructions explicitly authorize commits and final pushes
to this independent fork branch. No PR or upstream issue is requested. This
authorization overrides the inherited AGENTS instruction against agent pushes
for this owner's fork only; it does not authorize publishing upstream.
