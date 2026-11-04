"""Full causal output checks around the candidate sparse-prefill cutoff; no speed claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from freetoken.core import SamplingParams
from benchmarks.clover.harness import load_engine, save_results

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--corpus',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--cache-slots',type=int,required=True)
p.add_argument('--limit',type=int,choices=(128,256),required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
result=dict(state='running',cases=[],settings=dict(model=a.model,cache_slots=a.cache_slots,
    limit=a.limit,context=2048,output_limit=64,penalties='checkpoint defaults',embedding_host=True),
    corpus_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
    protocol='Six exact natural-text prefixes:128/129/192/256/257/384, two complete greedy '
    'causal requests each. Prefix reuse disabled, expert cache reset per request. '
    'Compare all visible output IDs, not only the first token. No throughput claim.')
llm=load_engine(a.model,context=2048,cache_slots=a.cache_slots,memory_ratio=.9,
    clover_max_width=0,clover_sparse_prefill_limit=a.limit,gguf_embedding_host=True)
try:
    cache=llm.engine.moe_offload_cache
    assert cache.cache_size==a.cache_slots
    assert llm.engine.config.host_embedding_enabled
    assert all(layer.mlp.sparse_prefill_limit==a.limit for layer in llm.engine.model.model.layers.op_list)
    match=llm.cache_manager.match_req
    llm.cache_manager.match_req=lambda req,max_len=None:match(req,max_len=0)
    rows=[json.loads(line) for line in a.corpus.read_text(encoding='utf-8').splitlines()]
    text='\n\n'.join(row['question']+'\n'+row['answer'] for row in rows[:80])
    seed=llm.tokenizer.apply_chat_template([dict(role='user',content='Summarize these worked examples.')],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    body=llm.tokenizer.encode(text,add_special_tokens=False)
    for length in (128,129,192,256,257,384):
        ids=seed[:-8]+body[:length-len(seed)]+seed[-8:]
        assert len(ids)==length
        for repeat in range(2):
            cache.reset()
            torch.cuda.synchronize()
            answer=llm.generate([ids],SamplingParams(temperature=0,max_tokens=64,clover_mode='causal'))[0]
            assert 'error' not in answer and 0<len(answer['token_ids'])<=64
            result['cases'].append(dict(input_length=length,prompt_ids=ids,repeat=repeat,**answer))
            save_results(a.output,result)
    result['state']='complete'
    save_results(a.output,result)
finally:
    llm.engine.shutdown()
