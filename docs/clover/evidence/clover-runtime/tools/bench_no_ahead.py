"""Diagnostic only: keep graphs/picking, recompute the pending token as before."""
import sys
sys.path.insert(0, 'C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken/benchmarks')
import bench_clover
from freetoken.models.clover_moe.graph import CloverBlockGraphs

original = CloverBlockGraphs.prepare_clean
def without_ahead(self, req, *, allow_ahead=False):
    return original(self, req, allow_ahead=False)
CloverBlockGraphs.prepare_clean = without_ahead
bench_clover.main()
