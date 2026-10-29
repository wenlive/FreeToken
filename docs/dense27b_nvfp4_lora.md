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

The owner's current instructions explicitly authorize commits and final pushes
to this independent fork branch. No PR or upstream issue is requested. This
authorization overrides the inherited AGENTS instruction against agent pushes
for this owner's fork only; it does not authorize publishing upstream.
