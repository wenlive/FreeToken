# Clover development records

Ongoing Clover development uses `feat/clover-gguf-q8` and the official Q8_0 GGUF.
As of 2026-11-04, `feat/clover-nvfp4` is retired at the owner's request; its final
maintenance notice is commit `8afcec6`. Its previous results remain historical records,
and unfinished NVFP4 work is not treated as qualified.

- [Current repair qualification and outstanding gates](qualification-20261104.md)
- [Performance index and metric definitions](clover-performance.md)
- [R0 runtime report](clover-runtime-performance.md)
- [R1 review and experiments](clover-runtime-review.md)
- [Current repair behavior and configuration](changes.md)
- [Benchmark commands and independent references](../../benchmarks/clover/README.md)
- [R0 evidence](evidence/clover-runtime/README.md)
- [R1 evidence](evidence/clover-runtime-review/README.md)
- [2026-11-04 repair evidence](evidence/20261104/README.md)

New Clover reports and evidence belong under this directory. Benchmark code is kept
separately from serving code. The existing R0/R1 archives were relocated byte for byte;
their inventories, raw commands, source patches and historical results are unchanged.
Their relative document links and inventory paths still resolve. Literal old repository
paths in recorded commands describe the source checkout used at measurement time.

To verify the relocated R1 archive from the current repository root:

```powershell
python docs/clover/evidence/clover-runtime-review/tools/verify_inventory.py
```

Both archives have an `inventory.json` mapping paths relative to their own directory to
byte length and SHA256. The relocation verified every listed digest in both archives.
Do not rewrite historical manifests or patches to make them look like current runs.

The 2026-11-04 implementation campaign is active. Its source-bound local runs are under
`C:/workspace/runs/T-20261104-clover-improve/`; the task log is
`C:/workspace/tasks/T-20261104-clover-improve/README.md`. Completed runs are archived above;
new snapshots are appended as qualification proceeds. Large binary operands remain in
the local run store, with their paths and hashes retained in the archive inventory.
