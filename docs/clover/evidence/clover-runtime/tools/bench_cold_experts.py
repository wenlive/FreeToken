"""Conservative CLI cache-scope control: reset native expert slots each request."""
import sys
sys.path.insert(0, 'C:/workspace/wt/T-20261102-clover-q8/FreeToken/benchmarks')
import torch
import bench_clover

original = bench_clover.LLM
def instrumented(*args, **kwargs):
    llm = original(*args, **kwargs)
    generate = llm.generate
    def cold(*a, **kw):
        llm.engine.moe_offload_cache.reset()
        torch.cuda.synchronize()
        return generate(*a, **kw)
    llm.generate = cold
    return llm
bench_clover.LLM = instrumented
bench_clover.main()
