"""Isolate W32 graph execution after restoring old pending/commit/pick controls."""
import argparse
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--eager-verify',action='store_true')
args,rest=p.parse_known_args()
sys.argv=[sys.argv[0],*rest]
sys.path.insert(0,'C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken/benchmarks')
import torch
import bench_clover
from freetoken.kernel.triton import clover_pick
from freetoken.models.clover_moe.graph import CloverBlockGraphs
from freetoken.models.clover_moe.preview import make_preview_batch
from freetoken.models.clover_moe.runtime import CloverRuntime

clean=CloverBlockGraphs.prepare_clean
verify=CloverRuntime._verify
def without_ahead(self,req,*,allow_ahead=False):
    return clean(self,req,allow_ahead=False)
def eager_preview(self,req,tokens):
    batch=make_preview_batch(self.engine,req,tokens,coreset_top_p=self.engine.config.clover_coreset_top_p)
    with self.engine.ctx.forward_batch(batch):
        return self.engine.model.forward()
def old_shape(self,req,tokens,capacity):
    inputs=tokens[:-1] if len(tokens)>1 else tokens
    if not args.eager_verify:
        return verify(self,req,inputs,capacity)
    batch=self._verify_batch(req,inputs,capacity=capacity)
    with self.engine.ctx.forward_batch(batch):
        return self.engine.model.forward()
def reference_pick(logits,state,block,mask_id):
    scores=logits.float().clone()
    scores[:,mask_id]=-float('inf')
    state.apply(scores,block,mask_id)
    picks=scores.argmax(-1)
    confidence=scores.softmax(-1).gather(-1,picks[:,None]).squeeze(-1)
    return picks.to(torch.int32),confidence
CloverBlockGraphs.prepare_clean=without_ahead
CloverRuntime._preview=eager_preview
CloverRuntime._verify=old_shape
clover_pick.pick_with_confidence=reference_pick
bench_clover.main()
