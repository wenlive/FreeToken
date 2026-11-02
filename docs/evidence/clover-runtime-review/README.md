# Clover R1 evidence

This batch supplements R0 without rewriting its files. Read the
[current report](../../clover-runtime-review.md) for interpretation and the
[metric glossary](../../clover-performance.md) before comparing rates.

`inventory.json` binds all files except itself by byte length and SHA256.
`.gitattributes` disables newline conversion. `registry.json` lists every recorded
experiment, source commit, timestamp, classification, exclusion reason and local
artifact location. A successful process is not automatically a correctness pass:
read its result scope and the report's outstanding C failures.

From the repository root, verify a fresh checkout with:

```powershell
python docs/evidence/clover-runtime-review/tools/verify_inventory.py
```

| Evidence family | Paths and scope |
|---|---|
| R0 noisy-control repeat | `runs/original-abab-v2-*`, `analyses/original-runtime-summary.json`: old runtime revert patch vs clean39a26c1, causal only |
| R1 core ABAB | `runs/{q8,nvfp4}-frozen-abab-*`, `analyses/*-frozen-summary.json`: immutable B/A/B/A sources, cap256/P1, fixed2560/4096slots |
| Instrumented denoise | `runs/q8-breakdown64-*`, `runs/nvfp4-block-costs64`, `analyses/*-budget-*.json`: cap64, call event envelopes; historical HF progress comes from R0 `correctness.json` |
| Dynamic expert payload | `runs/q8-phase-expert-traffic`: diagnostic counter snapshots per advance/preview/verify; phase logical bytes must sum to request total |
| Graph correctness | retained/default and split graph audits: all112 effective tails, two prefixes; padded full-W causal reference, not old one-row MMVQ |
| Capacity | auto lifecycle, host-embedding OFF/ON, W32 capacity and separate `q8-auto-w32-resource-*` whole-device NVML diagnostics |
| Long context | `q8-context-sweep`, `q8-paired-split-kv-v2`: exact1K/4K/16K/32K or paired16K/32K fixed inputs, capture excluded from replay medians; the first split candidate failed its numerical gate and is excluded |
| Numerical diagnosis | `q8-preview-ab-*-shared-state`: same inputs/weights and optional exported native clean KV/GDN; no timing claim, small targeted sample |
| Regressions | `*-review-regressions`, `mtp-*-review-*`, `qwen35-*-review`; real-checkpoint outputs compared with retained R0 main controls in `audit/final-audit.json` |
| Threshold correctness | `*-threshold-correctness-*`: paired before/after Count/JSON W8/W16 cap64, not an ABAB performance claim |
| Exclusions | failed attempts and development-source `q8-review-abab-1-after` remain classified in the registry; they are not pooled into accepted matrices |

Each `runs/<name>/manifest.json` includes the exact process command, import root,
measurement-time commit/dirty status, tracked patch hash, native `.pyd` hashes,
script receipts and device observations. Later manifests also include normalized
LF production-source hashes; earlier runs retain their tracked patches and
untracked Python sources. `source.patch` and `untracked-source/` reconstruct dirty
measurement code. Final report-only commits do not change measured arithmetic.
Core ABAB imports detached worktrees, not the development tree. The optional
embedding and split-KV experiments are separate source/configuration cohorts.

Full outputs, token IDs, arrival arrays, warmups and measured requests are kept in
`runs/<name>/data/results.json`; stdout/stderr and available test XML/logs are also
retained. Large tensor dumps, weights, dependencies and compiled binaries are not
embedded. Retained tensor locations, sizes and hashes are listed in the registry.
Absolute Windows paths identify local originals; adapt them when reproducing.

The hardware/boot snapshot is `audit/final-system.json`, dependencies and R0
inventory audit are in `audit/results.json`, and final weight/shared-source checks
are in `audit/final-audit.json`. The R0 official Q8 digest remains
`834529e7aae000305ae94656782726bcf0ed77976f95a56383447d8903afddab`.
No model file was modified. Q8 and NVFP4 have different weights and bank sizes;
the primary source comparisons are within each precision.

To reproduce core timings, use the pinned environment described in R0 provenance,
one GPU workload at a time, and four fresh processes in before/after/before/after
order. Each uses the same benchmark script and settings, for example:

```powershell
python benchmarks/bench_clover.py --model MODEL --output NEW_DIRECTORY --cache-slots 2560 --memory-ratio .98 --tokens 256 --warmups 2 --repeats 5
python benchmarks/summarize_clover_abab.py --turns B0/results.json A1/results.json B2/results.json A3/results.json --output summary.json
python benchmarks/analyze_clover_budget.py --native PROFILE/results.json --hf docs/evidence/clover-runtime/correctness.json --output NEW_BUDGET_DIRECTORY
```

Exact source roots, private extension-cache environment and all actual arguments
take precedence over this shortened example; see manifests and `tools/` scripts.
The auto-cache .9 runs declare a W32 reserve. Explicit .98 near-capacity controls
are experiments, not a general desktop-memory guarantee.
Do not combine cap64/128/256, P0/P1, instrumented/uninstrumented, hot static/dynamic
routing, PyTorch/NVML peaks, or end-to-end/arrival-decode/round-event throughput.
