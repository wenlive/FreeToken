"""Measure the actual Q8-sized mapped pinned bank allocation on Windows."""
import argparse
import json
from pathlib import Path
import time

import torch
from freetoken.moe.host_banks import HostBank
from freetoken.kernel.pinned import device_ptr

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
result = dict(target_bytes=34225520640, layers=[], state="running")
banks = []
started = time.monotonic()
try:
    for layer in range(40):
        bank = HostBank((855638016,), torch.uint8)
        bank.tensor.zero_()
        bank.pin()
        pointer = device_ptr(bank.tensor)
        banks.append(bank)
        row = dict(layer=layer, bytes=bank.nbytes, pinned=bank.residency.value,
                   device_alias=pointer != bank.tensor.data_ptr(), seconds=time.monotonic() - started)
        result["layers"].append(row)
        print(json.dumps(row), flush=True)
    result.update(state="passed", allocated_bytes=sum(b.nbytes for b in banks))
except Exception as exc:
    result.update(state="failed", error=repr(exc), allocated_bytes=sum(b.nbytes for b in banks))
    raise
finally:
    result["seconds"] = time.monotonic() - started
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
