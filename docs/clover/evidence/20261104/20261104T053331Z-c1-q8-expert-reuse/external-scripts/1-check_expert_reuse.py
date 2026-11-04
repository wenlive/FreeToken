"""Diagnostic-only per-preview expert identity/reuse, with the real captured forward."""
import json
from pathlib import Path
import sys

import torch

if (Path.cwd() / 'benchmarks/clover/bench_clover.py').exists():
    from benchmarks.clover import bench_clover
else:
    sys.path.insert(0, str(Path.cwd() / 'benchmarks'))
    import bench_clover

records = []
original_llm = bench_clover.LLM

def instrumented(*args, **kwargs):
    llm = original_llm(*args, **kwargs)
    cache, runtime = llm.engine.moe_offload_cache, llm.engine.mtp_runtime
    # The Torch scatter is included when the block graph is captured below.
    # Native decode graphs were already captured; they are not measured here.
    cache.collect_decode_freq = True
    block = [-1]
    denoise = runtime._denoise

    def observed_denoise(*a, **kw):
        block[0] += 1
        return denoise(*a, **kw)
    runtime._denoise = observed_denoise

    def instrument(name, phase):
        original = getattr(runtime, name)
        def observed(req, tokens, *a, **kw):
            cache.decode_freq.zero_()
            stats = cache.lru_stats.clone()
            before = cache.id_of_slot.clone()
            graphs = runtime._blocks()
            keys = set(graphs.entries)
            output = original(req, tokens, *a, **kw)
            counts = cache.decode_freq.cpu()
            assert counts.any(), 'Routing histogram did not execute inside the graph'
            records.append(dict(block=block[0], phase=phase, prefix=req.cached_len,
                rows=len(tokens), cache_slots=cache.cache_size,
                captured=set(graphs.entries) != keys,
                active_ids=counts.flatten().nonzero().flatten().tolist(),
                counts=counts.tolist(), resident_before=before.cpu().tolist(),
                resident_after=cache.id_of_slot.cpu().tolist(),
                stats_delta=(cache.lru_stats - stats).cpu().tolist()))
            return output
        setattr(runtime, name, observed)
    instrument('_preview', 'preview')
    instrument('_verify', 'verify')
    return llm

bench_clover.LLM = instrumented
try:
    bench_clover.main()
finally:
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        blocks = []
        for block in sorted({r['block'] for r in records}):
            rows = [r for r in records if r['block'] == block]
            previews = [r for r in rows if r['phase'] == 'preview']
            if not previews:
                continue
            first = set(previews[0]['active_ids'])
            slots = previews[0]['cache_slots']
            union = set().union(*(set(r['active_ids']) for r in previews))
            blocks.append(dict(block=block, previews=len(previews), first_union=len(first),
                cumulative_union=len(union), slots=slots, half_capacity=slots // 2,
                per_layer_first=[sum(v > 0 for v in layer) for layer in previews[0]['counts']],
                repeated_first_experts=[len(first.intersection(r['active_ids'])) for r in previews[1:]],
                missing_first_at_next_start=[len(first.difference(r['resident_before'])) for r in previews[1:]],
                verify_union=[len(r['active_ids']) for r in rows if r['phase'] == 'verify']))
        (output / 'expert-reuse.json').write_text(json.dumps(dict(scope='instrumented identity diagnostic; '
            'timings are not performance evidence; initial graph capture may multiply counts',
            blocks=blocks, forwards=records), indent=2), encoding='utf-8')
