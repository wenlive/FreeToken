# Clover measurements

Run from the repository root with `PYTHONPATH=python:.` and a local checkpoint.
The package keeps measurement and independent reference code outside serving.

```bash
python -m benchmarks.clover --help
python -m benchmarks.clover bench --help
python -m benchmarks.clover preview graphs --help
python -m benchmarks.clover hf-check denoise --help
```

| Entry | Default | Optional selector |
|---|---|---|
| `bench` | Five-prompt mode matrix, token arrivals and output IDs | `resources` for Windows NVML diagnostics |
| `context` | Exact long-prefix preview/verify timing | |
| `preview` | First-anchor preview latency | `split-kv`, `audit`, `graphs`, `lifecycle`, `verify`, `ab`, `outputs` |
| `costs` | Native phase costs with HF progress | `abab` for four chronological process results |
| `hf-check` | Same-weight official HF preview comparison | `denoise`, `selfspec`, `quantization` |
| `llama-compare` | Stock llama.cpp comparison | |
| `quality` | Resumeable GSM8K and HumanEval evaluation | `fetch-data` for explicit dataset preparation |

Each selected tool accepts `--help`. `fetch-data` accesses the network only when
explicitly invoked; other measurements consume prepared local inputs. The HumanEval
worker remains a separate isolated `python -I` subprocess with a timeout.

`harness.py` owns the fixed prompts, native engine defaults and JSON result writing.
`clover_metrics.py` handles visible token arrivals, excluding terminal EOS and the first
burst from decode throughput. `summarize_clover_abab.py` preserves process drift and both
chronological before/after ratios. Arithmetic references live in `oracles.py` and the
HF/GGUF reference modules; they are not performance backends.

Example Q8 causal measurement:

```bash
python -m benchmarks.clover bench --model /path/to/Clover-Q8_0.gguf --output NEW_DIRECTORY --cache-auto --memory-ratio .9 --modes causal --tokens 256 --warmups 2 --repeats 5
```

The matrix declares the largest selected block width for capacity planning; an explicit
`--clover-max-width` overrides it. Embedding placement follows engine defaults;
`--embedding-host` and `--no-embedding-host` explicitly select mapped host or device storage.
`--long-prompts-json FILE --prompts NAME` uses saved `prompts.NAME.user_text`; the result
records actual input IDs. Prefix reuse is disabled by default. CUDA event and cache
counter instrumentation (`--runtime-breakdown`, `--phase-cache-stats`) belongs in a
separate diagnostic run, because it adds overhead.

Real-model regressions live under `tests/` with `needs_weights` markers and local model
environment variables. The graph and lifecycle commands remain explicit hardware audits.
The retired long-prompt wrapper and CPU-only experiment are preserved byte for byte in
`docs/clover/evidence/legacy-tools/`; use their original source revision to reproduce old
commands. Current long-prompt measurements use the matrix entry above.

Results and historical provenance are indexed in [docs/clover](../../docs/clover/README.md).

`clover_llama_preview.cpp` also accepts `--clover-cases INPUT --clover-output DIRECTORY`
for fixed-history distribution diagnostics against the stock llama.cpp library. This
mode exports logits instead of running the draft-self timing loop. Create the output
directory first; existing output files are rejected. The whitespace-separated input is:

```text
number_of_cases
prefix_length causal_steps number_of_preview_blocks
prefix_token_ids...
forced_causal_token_ids...
preview_width
preview_token_ids...
```

Repeat width/tokens for each preview, and the case fields for each case. Causal steps
must be at least two; preview widths are 8, 16 or 32. Partial blocks may contain fixed
tokens in arbitrary positions. Each preview restores the same prefilled state. A final
causal replay must reproduce the saved logits byte for byte, or the program fails.
Files `case-N-causal.f32` and `case-N-preview-B.f32` contain row-major vocabulary logits
(little-endian float32 on the supported x86 hosts); the JSON stdout record supplies the
vocabulary size and restoration result. This does not implement a LowConfidence loop
and must not be reported as llama denoising throughput.
