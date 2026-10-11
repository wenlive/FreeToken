# GSQ Q2_0 validation

Feature commit: `dbabd024595b4154e37b7f1aa5e6c3718b599b5d`.
Base: `74a387777708b9650a027494bcb37d2ab9cf2443`.

These tests were executed by an AI assistant on the contributor's machine.
They are not a claim of a manual human rerun.

Windows 11, Ryzen 9 9900X, 64 GB RAM, RTX 5090 D v2 24 GB,
NVIDIA driver 610.88, Python 3.12.13.

With the feature checkout as the working directory and its `python` directory
on `PYTHONPATH`, the command in each dependency environment was:

```powershell
python -B -m pytest tests/moe/test_q2_0.py tests/moe/test_q2_source.py -q
```

- Torch 2.11.0+cu130 / triton-windows 3.6.0.post26: 16 passed (includes five CUDA cases).
- Torch 2.14.1+cpu: 11 passed, five CUDA cases skipped.

The GPU tests use independent Torch matvec/routing results and native packed
bytes, including signed/zero scales, cache-slot permutation and graph replay.
The reader tests write real small GGUF files and exercise order and invalid sources.

The complete real expert source from
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, was also read in bounded chunks:

```powershell
python -B validation/gsq-q2/read_complete_source.py --model <verified-hf-config-directory> --output complete-source.json
```

This verifies 48 layers x 512 experts and all 33,973,862,400 packed bytes.
It does not allocate the full expert bank or run a complete model.
The source's full SHA256 was verified during preparation; this command uses
that receipt and independently validates/reads the complete expert directory.

Linux, GPU execution with main's Torch 2.14.1 / Triton 3.8 dependency set,
and full-model generation on this exact upstream feature are pending.
There are no end-to-end performance claims.
