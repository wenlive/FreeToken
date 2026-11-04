# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

Clover branch measurements are indexed in [the performance record](../docs/clover/clover-performance.md).
`bench_clover.py` records visible outputs, TTFT, arrival-based decode and expert
payload. Use `--runtime-breakdown` only for a separate event-instrumented run;
`--phase-cache-stats` adds GPU counter snapshots and is also diagnostic.
`--cache-auto --memory-ratio .9 --clover-max-width 32` includes the private block
window in capacity planning; `--embedding-host` tests the optional mapped table.
`bench_clover_context.py` measures exact 1K/4K/16K/32K prefixes;
`bench_clover_split_kv.py` compares two read-only preview graphs on identical state.
`analyze_clover_budget.py` combines measured costs with recorded HF progress,
and `summarize_clover_abab.py` retains both chronological ratios and process drift
for four before/after/before/after result files. Their `--help` gives the arguments.
