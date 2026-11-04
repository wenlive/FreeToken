"""Paired ABAB whole-model preview graph replays at fixed long prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from clover.oracles import make_preview_batch
from check_clover_preview import persistent_hashes


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefixes',default='16384,32768')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    prefixes=list(map(int,a.prefixes.split(',')))
    llm=LLM(a.model,moe_strategy='offload',moe_cache_size=2304,expert_load='serial',
        max_running_req=1,memory_ratio=.95,clover_max_width=0,
        num_token_override=max(prefixes)+128,max_seq_len_override=max(prefixes)+128,
        max_extend_tokens=4096,cuda_graph_bs=[1])
    engine=llm.engine
    result=dict(state='running',scope='Same model/prefix/mask/KV/GDN state, two independent graphs, '
        'ABAB with two warmups plus five CUDA-event measurements per turn; no metadata/input staging in timed replay',cases=[])
    def save():
        (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    forward=engine.forward_batch
    current=None
    def audited(batch,sampling):
        output=forward(batch,sampling)
        req=batch.reqs[0]
        if batch.is_prefill and req.cached_len==current:
            for width in (8,16,32):
                tokens=torch.full((width,),engine.mtp_runtime.mask_id,device='cuda',dtype=torch.int32)
                frozen=persistent_hashes(engine)
                graphs=[]
                logits=[]
                for splits in (1,8):
                    temporary=make_preview_batch(engine,req,tokens)
                    temporary.clover_kv_splits=splits
                    with engine.ctx.forward_batch(temporary):
                        for _ in range(2):engine.model.forward()
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=engine.stream):
                        with engine.ctx.forward_batch(temporary):scores=engine.model.forward()
                    graph.replay()
                    logits.append(scores.float().cpu())
                    graphs.append((temporary,graph,scores))
                reference,actual=logits
                error=float((actual-reference).norm()/reference.norm())
                if not torch.isfinite(actual).all() or error >= .05:
                    result['state']='numerical_gate_failed'
                    result['failure']=dict(prefix=current,width=width,relative_l2=error,
                        finite=bool(torch.isfinite(actual).all()),
                        argmax_agreement=float((actual.argmax(-1)==reference.argmax(-1)).float().mean()))
                    save()
                assert torch.isfinite(actual).all() and error<.05
                row=dict(prefix=current,width=width,relative_l2=error,
                    argmax_agreement=float((actual.argmax(-1)==reference.argmax(-1)).float().mean()),turns=[])
                for turn in range(4):
                    graph=graphs[turn%2][1]
                    timings=[]
                    for _ in range(7):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record(engine.stream);graph.replay();end.record(engine.stream);end.synchronize()
                        timings.append(begin.elapsed_time(end))
                    row['turns'].append(dict(backend='split8' if turn%2 else 'standard',
                        measured_ms=timings[2:],warmup_ms=timings[:2],median_ms=statistics.median(timings[2:])))
                row['state_unchanged']=persistent_hashes(engine)==frozen
                assert row['state_unchanged']
                result['cases'].append(row)
                print(json.dumps(row),flush=True)
                save()
                graphs.clear()
        return output
    engine.forward_batch=audited
    match=llm.cache_manager.match_req
    llm.cache_manager.match_req=lambda req,max_len=None:match(req,max_len=0)
    try:
        seed=llm.tokenizer.apply_chat_template([dict(role='user',content='Describe a tree.')],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        pattern=llm.tokenizer.encode('Trees shade the streets. ')
        for current in prefixes:
            middle=current-len(seed)
            ids=seed[:-8]+(pattern*((middle+len(pattern)-1)//len(pattern)))[:middle]+seed[-8:]
            result.setdefault('inputs',[]).append(dict(prefix=current,token_ids=ids,
                sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
            llm.generate([ids],SamplingParams(temperature=0,max_tokens=1,clover_mode='causal'))
        result['state']='complete'
        save()
    finally:engine.shutdown()


if __name__=='__main__':main()
