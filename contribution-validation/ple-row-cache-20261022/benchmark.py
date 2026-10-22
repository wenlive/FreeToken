"""Measure DiskRowTable fill + H2D + dequantization, not whole-model throughput."""
import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--cache-rows', type=int, default=0)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f'Refusing to overwrite {args.output}')
sys.path[:0] = [str(args.source / 'python'), str(args.source)]

import torch
from freetoken.kernel import _row_store
from tests.models.qwen4_exp.test_ple import _meta
from tests.models.qwen4_exp.test_ple_disk import _bitwise_equal, _embedding, _make_table

torch.set_num_threads(1)
emb = _embedding()
generator = torch.Generator().manual_seed(271)
# Fixed-size decode batches; repeat a small context working set, then stream new contexts.
hot = torch.randint(0, 512, (16, 3), generator=generator, dtype=torch.int64)
streaming = torch.randint(0, 512, (160, 16, 3), generator=generator, dtype=torch.int64)
traces = {'repeated-contexts': hot.unsqueeze(0).expand(160, -1, -1), 'streaming-contexts': streaming}
results = []
with tempfile.TemporaryDirectory(prefix='ple-cache-bench-') as folder:
    for name, trace in traces.items():
        location = Path(folder) / name
        location.mkdir()
        disk, oracle, model_args = _make_table(location)
        if args.cache_rows:
            disk._store.set_row_cache(args.cache_rows)
        batches = []
        for windows in trace:
            runs = [window.clone() for window in windows]
            tokens = [[int(w[2])] for w in windows]
            contexts = [[int(w[0]), int(w[1])] for w in windows]
            ids = emb.row_ids(_meta(tokens, contexts, decode=True)).cuda()
            batches.append((runs, ids, oracle.lookup(ids)))
        samples = []
        stats_start = stats_end = None
        for trial in range(6):
            outputs = []
            torch.cuda.synchronize()
            if hasattr(disk._store, 'io_stats'):
                stats_start = disk._store.io_stats()
            start = time.perf_counter()
            for runs, ids, _ in batches:
                disk.fill(runs, graph=False)
                outputs.append(disk.lookup(ids))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if hasattr(disk._store, 'io_stats'):
                stats_end = disk._store.io_stats()
            assert all(_bitwise_equal(got, want) for got, (_, _, want) in zip(outputs, batches))
            if trial:
                samples.append(elapsed * 1e6 / len(batches))
        result = dict(workload=name, batch_size=16, steps=len(batches), trials=5,
                      us_per_batch=samples, median_us=statistics.median(samples),
                      stats_start=stats_start, stats_end=stats_end)
        results.append(result)
        del disk, oracle, batches
report = dict(source=str(args.source), extension=_row_store.__file__, torch=torch.__version__,
              gpu=torch.cuda.get_device_name(), cache_rows=args.cache_rows,
              scope='DiskRowTable.fill + GPU lookup, synthetic FP8 table, not whole-model tokens/s',
              results=results)
args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
