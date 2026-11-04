# Clover repair campaign evidence

This archive grows as qualification finishes. Each run retains its original manifest,
source patch, command, logs, external diagnostic scripts and JSON results. Process exit
success means a diagnostic completed; it does not by itself certify numerical agreement.
Failed reproductions and failed setup attempts are retained too.

`inventory.json` records the measured source commit, original path, byte count and SHA256.
Text artifacts above 4 MiB use deterministic gzip; their decompressed bytes exactly match
the originals. Tensor operands, synthetic binary test artifacts and the physical source
copy used by the extraction check remain in the local
run store named by `source_root`, with their paths and hashes recorded in the inventory.
Historical command paths are provenance and have not been rewritten.

From the repository root:

```powershell
python docs/clover/evidence/20261104/tools/verify_inventory.py
python docs/clover/evidence/20261104/tools/verify_inventory.py --tracked
python docs/clover/evidence/20261104/tools/verify_inventory.py --verify-external
```

`--tracked` also checks that Git retains exactly those bytes, including normally ignored
logs. `--verify-external` requires this campaign's local run store. Early unwrapped runs
are identified explicitly; their source identity is not invented. New snapshots append
run files and update the inventory; existing run bytes must remain identical.
