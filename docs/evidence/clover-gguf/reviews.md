# Staged review record

These are self-reviews of the task's implementation and evidence, not an
independent external review. Full notes and immutable command logs remain under
`C:/workspace/tasks/T-20261031-clover-gguf/review-*.md` and the task's run folder.

| Block | Finding and resulting improvement | Validation |
|---|---|---|
| Foundation | Preserve four-row Flash-Next defaults, lazy model runtime dispatch, prefix ownership and Apache source headers | Targeted contracts, then real main/feature MTP regression |
| GGUF adapter | Invert GDN head permutations, unfold norms in FP32, retain all original special-token IDs, reject unsupported layouts early | 583 buffers,421 source tensors,248077 IDs, actual model smoke |
| Arithmetic | Clover requires FP32 down/router multiply, BF16 per-route rounding and FP32 sum; generic Q4_0 rounds earlier | Separate Clover helper; independent Q8_1 dot oracle and complete routed tests; old Q4_0/NVFP4 unchanged |
| Measurement | Windows venv launcher RSS was invalid; observer must include model subprocesses | New full matrix with recursive RSS/private/system-available samples; failed record retained |
| Shared paths | Keep old DeliveryLimit import compatible and avoid changing existing numerical dispatch | Added-line lint audit; pools, scheduler bundle tests and actual Flash-Next off/on |
| HF oracle | NumPy decode repeated work and took938.9s | Independent eager Torch decoder, bitwise NumPy CPU/GPU checks and18 actual projections;128.0s full trace reproduces old decisions |
| AOT | Full regression exposed missing Clover GGUF architecture alias and mixed row sizes | Actual bank-byte contracts, targeted19 passes, full403 passes/147 skips/47 subtests |
| Official comparison | Input LF stripping, EOS overcount, SSE duplicates, metadata penalty conflict and unsupported server entropy flags | Five accounting tests plus real stock CLI/server smokes; separate protocols and cache scopes |
| Evidence | First selfspec round occurs after prefill; only denoise should omit its prefill-containing first block | Corrected derived cost model without altering raw records; SHA bindings, all25progress gates and readonly hashes asserted |
| Final comparison | Ordinary loading helps Python but is not a universal gain; Count has19.4% temporal drift while other native prompts vary0.2%-2.2% | Eight completed comparison datasets/436requests, all host observers complete, exact input checks,11/15 P0 output pairs identical |
| Output validity | Algorithm progress does not establish generated program correctness | Native causal/SS Python pass; all3denoise Python fail; all6GGUF JSON outputs pass; stock measured profiles' Python/JSON pass |

Known limits remain explicit: three causal Chinese rank reversals; selfspec
and HF text differences;64-token block-reference horizon; first50 quality
subsets; optional/full-Qwen4-HF skips; no physical PCIe profiler attribution.
Neither fused NVFP4 numerical source file changes relative to main. Source tree
`0e8787db0be5c049f7d6dd0c48aaa340a8c3c538` is bound to the qualification.
