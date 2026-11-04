"""Capture real first-block LowConfidence inputs for a matched-trajectory numerical ruler."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from benchmarks.clover.harness import PROMPTS, load_engine
from freetoken.core import SamplingParams
from freetoken.kernel.triton import clover_pick
from freetoken.models.clover_moe import runtime as runtime_module

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    llm=load_engine(a.model,context=4096,cache_slots=2292,memory_ratio=.9,
        clover_max_width=32,clover_sparse_prefill_limit=128,gguf_embedding_host=False)
    engine,runtime=llm.engine,llm.engine.mtp_runtime
    match=llm.cache_manager.match_req
    llm.cache_manager.match_req=lambda req,max_len=None:match(req,max_len=0)
    ids=llm.tokenizer.apply_chat_template([dict(role='user',content=PROMPTS['JSON'])],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    case=dict(prompt='JSON',prompt_ids=ids,mask_id=runtime.mask_id,blocks=[],trajectory=[])
    scores,previews,sampled=[],[],[]
    phase='causal'
    old_logits,old_sample=runtime.process_logits,engine.sampler.sample
    old_preview,old_pick,old_resolve=runtime._preview,clover_pick.pick_with_confidence,runtime_module.resolve_low_confidence
    def logits(batch,value):
        if phase=='causal':
            scores.append(value[0].float().cpu())
        return old_logits(batch,value)
    def sample(value,args):
        tokens=old_sample(value,args)
        if phase=='causal':
            sampled.append(int(tokens[0]))
        return tokens
    def preview(req,tokens):
        case['blocks'].append(dict(name=f'w32-native-step-{len(previews):02d}',ids=tokens.cpu().tolist()))
        value=old_preview(req,tokens)
        previews.append(value.float().cpu())
        return value
    def pick(value,state,block,mask):
        prediction,confidence=old_pick(value,state,block,mask)
        assert state.length==0 and int(state.counts.count_nonzero())==0
        # Prove that first-block penalties other than the mask do not affect this fixture.
        adjusted=value.float().clone()
        state.apply(adjusted,block,mask)
        masked=value.float().clone()
        masked[:,mask]=-float('inf')
        torch.testing.assert_close(adjusted,masked,rtol=0,atol=0)
        oracle=masked.softmax(-1)
        torch.testing.assert_close(prediction,oracle.argmax(-1),check_dtype=False,rtol=0,atol=0)
        torch.testing.assert_close(confidence,oracle.amax(-1),rtol=2e-6,atol=2e-7)
        case['trajectory'].append(dict(input_ids=block.cpu().tolist(),prediction=prediction.cpu().tolist(),
            confidence=confidence.cpu().tolist()))
        return prediction,confidence
    def resolve(preview_fn,state,**kwargs):
        block,steps=old_resolve(preview_fn,state,**kwargs)
        case.update(final_block=block.cpu().tolist(),steps=steps)
        return block,steps
    runtime.process_logits,engine.sampler.sample=logits,sample
    runtime._preview,clover_pick.pick_with_confidence,runtime_module.resolve_low_confidence=preview,pick,resolve
    try:
        answer=llm.generate([ids],SamplingParams(temperature=0,max_tokens=32,ignore_eos=True,clover_mode='causal'))[0]
        assert len(sampled)>=32 and len(scores)>=32
        case.update(forced_ids=sampled[:32],visible_ids=answer['token_ids'])
        phase='denoise'
        answer=llm.generate([ids],SamplingParams(temperature=0,max_tokens=32,clover_mode='denoise',clover_width=32))[0]
        case['denoise_visible_ids']=answer['token_ids']
        assert len(previews)==len(case['steps'])==len(case['trajectory'])
        for index,row in enumerate(case['trajectory']):
            pending=[i for i,t in enumerate(row['input_ids']) if t==runtime.mask_id]
            selected=[i for i in pending if row['confidence'][i]>.95]
            if not selected:
                selected=[max(pending,key=lambda i:row['confidence'][i])]
            expected=row['input_ids'][:]
            for position in selected:
                expected[position]=row['prediction'][position]
            following=case['trajectory'][index+1]['input_ids'] if index+1<len(case['trajectory']) else case['final_block']
            assert expected==following and len(selected)==case['steps'][index]
            row['selected']=selected
        artifact=a.output/'json-native.pt'
        torch.save(dict(causal=torch.stack(scores[:32]),previews=previews),artifact)
        case['artifact']=dict(file=artifact.name,sha256=digest(artifact))
        lines=['1',f'{len(ids)} 32 {len(previews)}',' '.join(map(str,ids)),' '.join(map(str,case['forced_ids']))]
        for block in case['blocks']:
            lines+=['32',' '.join(map(str,block['ids']))]
        cases=a.output/'cases.txt'
        cases.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        result=dict(state='complete',stage='native',model=a.model,hf_source=a.source,steps=32,cases=[case],
            cases_text_sha256=digest(cases),protocol='Real native JSON W32 first-block trajectory, checkpoint penalties. '
            'First-block non-mask penalties verified inert; fused picks/confidence checked against Torch. '
            'Causal forced history is recorded for the stock llama checkpoint/replay protocol. No timing claim.')
        (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(dict(steps=case['steps'],previews=len(previews),visible_ids=answer['token_ids'])))
    finally:
        runtime.process_logits,engine.sampler.sample=old_logits,old_sample
        runtime._preview,clover_pick.pick_with_confidence,runtime_module.resolve_low_confidence=old_preview,old_pick,old_resolve
        engine.shutdown()

if __name__=='__main__':
    main()
