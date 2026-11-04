"""Run the new high-slot oracle against the pre-fix header in an isolated build."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from freetoken.kernel import gguf

repo = Path("C:/workspace/wt/T-20261102-clover-q8/FreeToken")
destination = Path("C:/workspace/builds/T-20261102-clover-q8/mmq-before-slot-fix-csrc")
shutil.copytree(repo / "python/freetoken/kernel/csrc/gguf", destination)
old = subprocess.run(["git", "show", "25fb44b:python/freetoken/kernel/csrc/gguf/moe.cuh"],
    cwd=repo, check=True, capture_output=True).stdout
(destination / "moe.cuh").write_bytes(old)
assert b"exp_idx > 255" in old
gguf._CSRC = destination
gguf._module.cache_clear()
sys.exit(pytest.main(["-q", "tests/kernels/test_clover_gguf.py::test_mmq_large_cache_slots_and_byte_offsets_above_2gib[34816]"]))
