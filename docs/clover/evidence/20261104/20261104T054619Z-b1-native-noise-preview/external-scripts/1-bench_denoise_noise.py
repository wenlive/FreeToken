"""Only alter attention prefix tile size, in selected phases, to measure noise floor."""
import argparse
from pathlib import Path
import sys

if (Path.cwd() / 'benchmarks/clover/bench_clover.py').exists():
    from benchmarks.clover import bench_clover
else:
    sys.path.insert(0, str(Path.cwd() / 'benchmarks'))
    import bench_clover
from freetoken.core import get_global_ctx
from freetoken.kernel.triton import attention

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--noise-phase', choices=['none', 'prefill', 'preview', 'both'], required=True)
a, rest = p.parse_known_args()
sys.argv = [sys.argv[0], *rest]
select = attention._select_extend_tile

def perturbed(*args):
    m, n = select(*args)
    batch = get_global_ctx().batch
    preview = getattr(batch, 'clover_preview', False)
    prefill = batch.is_prefill
    change = ((a.noise_phase in ('prefill', 'both') and prefill)
        or (a.noise_phase in ('preview', 'both') and preview))
    if change:
        if n <= 16:
            raise ValueError('Baseline tile too small for this noise experiment')
        return m, n // 2
    return m, n

attention._select_extend_tile = perturbed
try:
    bench_clover.main()
finally:
    attention._select_extend_tile = select
