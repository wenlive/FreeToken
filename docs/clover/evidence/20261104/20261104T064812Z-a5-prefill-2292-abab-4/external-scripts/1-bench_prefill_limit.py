"""Paired short-prefill crossover with exact natural-text inputs and fixed slots."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from freetoken.core import SamplingParams
from freetoken.message import DetokenizeMsg
from benchmarks.clover.harness import load_engine, save_results

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--corpus', type=Path, required=True)
p.add_argument('--cache-slots', type=int, required=True)
p.add_argument('--limit', type=int, choices=[128, 1024], required=True)
p.add_argument('--prefixes', default='129,192,256,384,512,768,1024')
p.add_argument('--warmups', type=int, default=2)
p.add_argument('--repeats', type=int, default=5)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
result = dict(state='running', settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
    protocol='Exact natural-text prefixes; prefix reuse disabled and expert cache reset each request. '
    'One visible token: decode throughput is not defined. Uninstrumented primary runs; one extra CUDA-event run per length.',
    corpus_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(), inputs=[], runs=[], forwards=[])
start = time.perf_counter()
llm = load_engine(a.model, context=2048, cache_slots=a.cache_slots, memory_ratio=.9,
    clover_max_width=0, clover_sparse_prefill_limit=a.limit)
result['startup_seconds'] = time.perf_counter() - start
cache, engine = llm.engine.moe_offload_cache, llm.engine
assert cache.cache_size == a.cache_slots
match = llm.cache_manager.match_req
llm.cache_manager.match_req = lambda req, max_len=None: match(req, max_len=0)
arrivals = []
send = llm.send_result

def receive(messages):
    stamp = time.perf_counter()
    arrivals.extend(stamp for m in messages if isinstance(m, DetokenizeMsg))
    send(messages)
llm.send_result = receive
forward = engine.forward_batch
measure = False
events = []

def observed(batch, params):
    if not measure:
        return forward(batch, params)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(engine.stream)
    output = forward(batch, params)
    end.record(engine.stream)
    events.append((begin, end))
    return output
engine.forward_batch = observed
try:
    rows = [json.loads(line) for line in a.corpus.read_text(encoding='utf-8').splitlines()]
    text = '\n\n'.join(row['question'] + '\n' + row['answer'] for row in rows[:80])
    seed = llm.tokenizer.apply_chat_template([dict(role='user', content='Summarize these worked examples.')],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    body = llm.tokenizer.encode(text, add_special_tokens=False)
    for length in map(int, a.prefixes.split(',')):
        ids = seed[:-8] + body[:length-len(seed)] + seed[-8:]
        assert len(ids) == length
        result['inputs'].append(dict(length=length, token_ids=ids))
        for repeat in range(a.warmups + a.repeats + 1):
            measure = repeat == a.warmups + a.repeats
            cache.reset()
            torch.cuda.synchronize()
            arrivals.clear()
            events.clear()
            begin = time.perf_counter()
            answer = llm.generate([ids], SamplingParams(temperature=0, max_tokens=1, clover_mode='causal'))[0]
            torch.cuda.synchronize()
            wall = time.perf_counter() - begin
            assert len(answer['token_ids']) == 1 and 'error' not in answer
            if measure:
                result['forwards'].append(dict(length=length, milliseconds=[x.elapsed_time(y) for x,y in events],
                    token_ids=answer['token_ids']))
            else:
                result['runs'].append(dict(length=length, repeat=repeat, warmup=repeat<a.warmups,
                    token_ids=answer['token_ids'], ttft_seconds=arrivals[0]-begin,
                    wall_seconds=wall, end_to_end_tokens_per_second=1/wall, decode_tokens_per_second=None))
            save_results(a.output, result)
    result['state'] = 'complete'
    save_results(a.output, result)
finally:
    engine.shutdown()
