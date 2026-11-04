"""Causal scheduling with concurrent, chunked, stopped and prefix-reused requests."""
import argparse
import json
from pathlib import Path

import torch
from freetoken.core import SamplingParams
from freetoken.env import ENV
from freetoken.llm import LLM

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
llm = LLM(a.model, moe_strategy='offload', moe_cache_auto=True, memory_ratio=.9,
    kv_reserve_tokens=4096, num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=128, max_running_req=2, clover_max_width=0,
    expert_load='serial', cuda_graph_bs=[1, 2])
result = dict(state='running', runs=[], penalties='checkpoint defaults',
    max_running_req=2, max_extend_tokens=128)
old = ENV.DISABLE_OVERLAP_SCHEDULING.value
try:
    prompts = []
    for text in ('Count from 1 to 10.', 'Write a short Python Fibonacci function.'):
        prompts.append(llm.tokenizer.apply_chat_template([dict(role='user', content=text)],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False))
    seed = llm.tokenizer.encode('Trees shade streets. ')
    long_prompt = prompts[1][:-8] + (seed * 150)[:742] + prompts[1][-8:]
    cases = [('two', prompts, [SamplingParams(clover_mode='causal', max_tokens=48)] * 2),
        ('chunked', [long_prompt, prompts[0]], [SamplingParams(clover_mode='causal', max_tokens=32)] * 2),
        ('stop', prompts, [SamplingParams(clover_mode='causal', max_tokens=48, stop_strs=['3']),
            SamplingParams(clover_mode='causal', max_tokens=32)])]
    # ABAB also exercises reuse of prefix snapshots and freed request tables.
    for disabled in (True, False, True, False):
        ENV.DISABLE_OVERLAP_SCHEDULING.value = disabled
        for name, ids, params in cases:
            answer = llm.generate(ids, params)
            torch.cuda.synchronize()
            llm.cache_manager.check_integrity()
            assert all(r['token_ids'] and not r.get('error') for r in answer)
            result['runs'].append(dict(case=name, overlap=not disabled, answers=answer))
            (a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, _, _ in cases:
        rows = [r['answers'] for r in result['runs'] if r['case'] == name]
        assert all(row == rows[0] for row in rows), name
    result['state'] = 'passed'
    (a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
finally:
    ENV.DISABLE_OVERLAP_SCHEDULING.value = old
    llm.engine.shutdown()
