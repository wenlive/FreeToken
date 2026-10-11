import argparse
from pathlib import Path
import sys

import freetoken.kernel
import pytest
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--native-dir", type=Path, required=True)
args = parser.parse_args()
freetoken.kernel.__path__.insert(0, str(args.native_dir.resolve()))
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
sys.exit(pytest.main(["tests/kernels/test_row_store.py", "tests/models/qwen4_exp/test_ple_disk.py", "-q", "-ra"]))
