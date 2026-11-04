"""Isolated multi-request penalty proposal; it does not alter the serving implementation."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from freetoken.kernel.triton.clover_verify import _adjust, _parameters, penalize_logits
from freetoken.models.clover_moe.generation import CloverPenaltyState, PENALTY_DEFAULTS


@triton.jit
def _request_penalties(LOGITS, OUT, META, STRIDE: tl.constexpr, V: tl.constexpr,
        MASK: tl.constexpr, PARAMS: tl.constexpr, TILE: tl.constexpr):
    row=tl.program_id(0)
    token=tl.program_id(1)*TILE+tl.arange(0,TILE)
    counts=tl.load(META+row*2).to(tl.pointer_type(tl.int64))
    length=tl.load(META+row*2+1)
    value=tl.load(LOGITS+row*STRIDE+token,token<V,other=-float('inf')).to(tl.float32)
    # Each row belongs to a distinct request, so hypothetical row offset is zero.
    value=_adjust(value,token,0,counts,LOGITS,length,MASK,V,True,False,PARAMS)
    tl.store(OUT+row*V+token,value,token<V)


def sequential(logits,states,mask):
    out=torch.empty(logits.shape,device=logits.device,dtype=torch.float32)
    for row,state in enumerate(states):
        penalize_logits(logits[row:row+1],state,mask,output=out[row:row+1])
    return out


def batched(logits,states,mask):
    if len(states)==1:
        return penalize_logits(logits,states[0],mask)
    assert all(state.active and state.gen is states[0].gen and state.stop_ids==states[0].stop_ids for state in states)
    # Fresh pinned storage avoids overwriting metadata still in flight. This probe
    # retains every state throughout measurement; production stream/lifetime tests
    # would be required before exposing an indirect-pointer API in serving.
    host=torch.tensor([(state.counts.data_ptr(),state.length) for state in states],
        dtype=torch.int64,pin_memory=True)
    packet=host.to(device=logits.device,non_blocking=True)
    out=torch.empty(logits.shape,device=logits.device,dtype=torch.float32)
    _request_penalties[(len(states),triton.cdiv(logits.shape[1],1024))](logits,out,packet,
        logits.stride(0),logits.shape[1],mask,_parameters(states[0]),1024,enable_fp_fusion=False)
    return out


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=200)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(state='running',cases=[],scope='GPU operator prototype only, no model or serving changes. '
        'Includes pinned metadata allocation/copy and output allocation. Two native and two candidate '
        'turns in ABAB order per batch size; no end-to-end generation speed claim.')
    gen=SimpleNamespace(**{**PENALTY_DEFAULTS,'presence_penalty':.9,'frequency_penalty':.2,
        'repetition_penalty':1.05,'dllm_length_penalty':10.,'dllm_length_penalty_start':8192,
        'dllm_length_penalty_window':8192,'dllm_think_end_bias':10.,
        'dllm_think_end_start':4096,'dllm_think_end_window':4096,'dllm_think_end_token_id':123})
    vocab=248320
    torch.manual_seed(42)
    for count in (1,2,4,8):
        states=[CloverPenaltyState(gen,vocab,'cuda',[vocab-3,vocab-2]) for _ in range(count)]
        for row,state in enumerate(states):
            state.commit(torch.tensor([7,7,11+row,*([123] if row%2 else [])],device='cuda'))
            state.length=4095+row*1024
        logits=torch.randn(count,vocab,device='cuda',dtype=torch.bfloat16)
        oracle=logits.float()
        oracle[:,-1]=-float('inf')
        frozen=[state.counts.clone() for state in states]
        for row,state in enumerate(states):
            state.apply(oracle[row:row+1],torch.tensor([vocab-1],device='cuda'),vocab-1)
        before=sequential(logits,states,vocab-1)
        after=batched(logits,states,vocab-1)
        torch.testing.assert_close(before,oracle,rtol=0,atol=0)
        torch.testing.assert_close(after,oracle,rtol=0,atol=0)
        assert all(torch.equal(state.counts,value) for state,value in zip(states,frozen))
        turns=[]
        for name,operation in (('sequential',sequential),('batched',batched),('sequential',sequential),('batched',batched)):
            for _ in range(20):
                operation(logits,states,vocab-1)
            torch.cuda.synchronize()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            started=time.perf_counter()
            begin.record()
            for _ in range(a.iterations):
                output=operation(logits,states,vocab-1)
            enqueue_done=time.perf_counter()
            end.record()
            end.synchronize()
            finished=time.perf_counter()
            turns.append(dict(variant=name,iterations=a.iterations,
                host_enqueue_us=(enqueue_done-started)*1e6/a.iterations,
                wall_us=(finished-started)*1e6/a.iterations,
                event_us=begin.elapsed_time(end)*1000/a.iterations))
        result['cases'].append(dict(requests=count,vocab=vocab,oracle_exact=True,counts_unchanged=True,
            turns=turns,batched_over_sequential_wall_pairs=[turns[i+1]['wall_us']/turns[i]['wall_us'] for i in (0,2)]))
        (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    result['state']='complete'
    (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
