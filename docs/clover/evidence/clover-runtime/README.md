# Clover Q8 and runtime evidence

This bundle is generated from completed local experiments. `inventory.json`
binds every bundled file except itself by SHA256. `sources.json` binds the
original files, exact commands, source commits, dirty patches and sampled host
memory. Absolute paths locate retained local runs; adapt them when reproducing
on another machine. No weights or compiled binaries are embedded.
The local `.gitattributes` disables newline conversion so inventory digests also
match a fresh checkout on another operating system.

- `comparison.json`: primary T1-T4 using unmodified cb09 stock and uniform
  `--load-mode none`. Exact and entropy cache scopes differ as the report states.
  Threshold/entropy uses the extra native per-request explicit cold expert-cache
  control. T3 additionally reports the slower of native hot/cold divided by the
  fastest stock width/rule, rather than hiding the cache-scope difference.
- `matrices.json`: all warmups and measured requests, summary medians, routing
  counters, round traces and original loading A/B. Stock strict is explicitly
  derived from the causal and selfspec datasets; its parents retain their commands.
- `outputs.json`: full text and token IDs deduplicated across matrices. Each
  request's `output_sha256` hashes UTF-8 JSON `[token_ids,text]` with compact
  separators and `ensure_ascii=False`. CLI has output text and producer token
  counts, without invented token IDs. Stream arrival arrays alone are omitted.
- `correctness.json`, `hf-initial-preview.json`: state hashes, independent
  same-Q8 HF oracle and progress differences, including unmet tolerances.
- `regression.json`, `quality.json`, `behavior.json`, `output-stability.json`:
  test coverage/skips, real checkpoint comparisons, all first-50 answers and
  grading failures, and changed denoise outputs. Scores are small-sample checks.
- `resources.json`: independent instrumented requests, whole-device NVML peak,
  physical PCIe sampling estimates and process-tree host memory. Counter window
  is 20ms; achieved polling gaps are recorded separately. Logical expert payload
  bytes exclude full-layer dense prefill uploads and are not physical bus bytes.
- `long-prompts.json`, `long-prefill-ab.json`: exact 1K/4K inputs and TTFT,
  plus larger stock microbatch capacity/performance probes without hiding failure.
- `numerical-diagnostics.json`, `nvfp4-capacity.json`: isolated changes to pending
  state, verification shape, pick arithmetic, and optional larger causal cache.
- `provenance.json`, `loading-ab.json`, `excluded-runs.json`: official weight
  origin/digest, adapter contracts, pinned-memory qualification, loading controls,
  preserved failures and invalid pre-MMQ-fix runs excluded from performance.

Reproduction helpers are in `tools/`; production benchmark tools are in the
repository's `benchmarks/`. `patches/` contains exact tracked changes for dirty
controls. Original full logs and tensor dumps remain under the recorded local
run paths. Source provenance is measurement-time provenance; later report-only
commits do not change measured runtime mathematics.

`logs/` additionally carries the MMQ regression, expected pre-fix numerical
failure, and official adapter/kernel test logs for direct reviewer inspection.
