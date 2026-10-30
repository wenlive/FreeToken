# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**Clover qualification** — `bench_clover.py` measures the five fixed prompts with
causal, selfspec and LowConfidence decoding, TTFT, stream arrivals and expert
payloads. `bench_clover_cpu.py` compares actual-bank CPU W4A16 with cold GPU fetch.
The `check_clover_hf_*.py` scripts use the pinned official arithmetic with bounded
on-demand expert storage; they are correctness oracles, not serving benchmarks.
`check_clover_quantization.py` compares NVFP4/FP8 on BF16-generated causal histories.
`eval_clover_quality.py` grades fixed GSM8K/HumanEval subsets; fetch only the
datasets with `fetch_clover_eval_data.py`. All model paths must already exist.
See [the Clover guide](../docs/clover-nvfp4.md) for settings, measurements and limits.

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
