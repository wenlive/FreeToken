"""Isolate batch-shape rounding from saved-ahead state and fused pick arithmetic."""
import argparse
import sys

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--reference-pick', action='store_true')
args, remaining = p.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
sys.path.insert(0, 'C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken/benchmarks')
import torch
import bench_clover
from freetoken.models.clover_moe.graph import CloverBlockGraphs
from freetoken.models.clover_moe.runtime import CloverRuntime

clean = CloverBlockGraphs.prepare_clean
verify = CloverRuntime._verify
def without_ahead(self, req, *, allow_ahead=False):
    return clean(self, req, allow_ahead=False)
def old_shape(self, req, tokens, capacity):
    # Denoise ignores these logits; only the first len(tokens)-1 state rows are
    # committed. The saved-ahead final row is deliberately never consumed here.
    return verify(self, req, tokens[:-1] if len(tokens) > 1 else tokens, capacity)
CloverBlockGraphs.prepare_clean = without_ahead
CloverRuntime._verify = old_shape
if args.reference_pick:
    from freetoken.kernel.triton import clover_pick
    def reference(logits, state, block, mask_id):
        scores = logits.float().clone()
        scores[:, mask_id] = -float('inf')
        state.apply(scores, block, mask_id)
        picks = scores.argmax(-1)
        confidence = scores.softmax(-1).gather(-1, picks[:, None]).squeeze(-1)
        return picks.to(torch.int32), confidence
    clover_pick.pick_with_confidence = reference
bench_clover.main()
