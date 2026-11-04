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
| `preview` | First-anchor preview latency | `split-kv`, `audit`, `graphs`, `lifecycle`, `verify`, `outputs` |
| `costs` | Native phase costs with HF progress | `abab` for four chronological process results |
| `hf-check` | Same-weight official HF preview comparison | `denoise`, `selfspec`, `quantization` |
| `quality` | Resumeable GSM8K and HumanEval evaluation | `fetch-data` for explicit dataset preparation |

Each selected tool accepts `--help`. `fetch-data` accesses the network only when
explicitly invoked; other measurements consume prepared local inputs. The HumanEval
worker remains a separate isolated `python -I` subprocess with a timeout.

`harness.py` owns the fixed prompts, native engine defaults and JSON result writing.
`clover_metrics.py` handles visible token arrivals, excluding terminal EOS and the first
burst from decode throughput. `summarize_clover_abab.py` preserves process drift and both
chronological before/after ratios. Arithmetic references live in `oracles.py` and the
HF reference module; they are not performance backends.

Example causal measurement:

```bash
python -m benchmarks.clover bench --model /path/to/clover-nvfp4 --output NEW_DIRECTORY --cache-auto --memory-ratio .9 --modes causal --tokens 256 --warmups 2 --repeats 5
```

The matrix declares the largest selected block width for capacity planning; an explicit
`--clover-max-width` overrides it. `--embedding-host` evaluates the mapped embedding table.
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
