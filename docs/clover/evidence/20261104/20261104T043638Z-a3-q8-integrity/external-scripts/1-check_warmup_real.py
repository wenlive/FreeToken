"""Cold block latency and optional byte-level startup/rebuild warmup audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.message import DetokenizeMsg
from freetoken.models.clover_moe import warmup

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--mode', choices=['selfspec', 'denoise'], default='selfspec')
p.add_argument('--width', type=int, default=16)
p.add_argument('--warmup', action='store_true')
p.add_argument('--audit', action='store_true')
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
result = dict(state='running', settings=vars(a).copy(), runs=[], audits=[])
result['settings']['output'] = str(a.output)

def save():
    (a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def snapshot(runtime, manager, tables):
    engine = runtime.engine
    pool = engine.linear_state_pool
    values = [engine.page_table, pool.conv_states, pool.recurrent_states]
    for layer in engine.model.model.layers.op_list:
        if not layer._is_linear:
            values.extend([engine.kv_cache.k_cache(layer.self_attn.layer_id),
                engine.kv_cache.v_cache(layer.self_attn.layer_id)])
    return dict(hashes=[digest(v) for v in values], pages=manager.available_size,
        states=pool.num_free_slots, tables=tables.available_size,
        prefix_pages=manager.prefix_cache.full_evictable_size,
        prefix_states=manager.prefix_cache.mamba_evictable_size)

original = warmup.warmup_block_graphs
def audited(runtime, manager, tables):
    before = snapshot(runtime, manager, tables)
    outcome = original(runtime, manager, tables)
    after = snapshot(runtime, manager, tables)
    manager.check_integrity()
    result['audits'].append(dict(before=before, after=after, warmup=outcome))
    save()
    assert before == after, 'Warmup changed cached state, table contents or free capacity'
    return outcome

if a.audit:
    warmup.warmup_block_graphs = audited
begin = time.perf_counter()
llm = LLM(a.model, moe_strategy='offload', moe_cache_auto=True, memory_ratio=.9,
    kv_reserve_tokens=4096, num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, max_running_req=1, clover_max_width=32,
    clover_graph_warmup_widths=[8, 16, 32] if a.warmup else [],
    expert_load='serial', cuda_graph_bs=[1])
result['startup_seconds'] = time.perf_counter() - begin
runtime, cache = llm.engine.mtp_runtime, llm.engine.moe_offload_cache
result['warmup'] = getattr(runtime, 'graph_warmup', None)
result['cache_slots'] = cache.cache_size
match = llm.cache_manager.match_req
llm.cache_manager.match_req = lambda req, max_len=None: match(req, max_len=0)
arrivals = []
send = llm.send_result
def receive(messages):
    now = time.perf_counter()
    arrivals.extend(now for m in messages if isinstance(m, DetokenizeMsg))
    send(messages)
llm.send_result = receive

def keys():
    blocks = getattr(runtime, 'block_graphs', None)
    return sorted(blocks.entries) if blocks is not None else []

try:
    ids = llm.tokenizer.apply_chat_template([dict(role='user', content='Count from 1 to 10.')],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    for index in range(a.repeats):
        cache.reset()
        torch.cuda.synchronize()
        arrivals.clear()
        before = keys()
        begin = time.perf_counter()
        answer = llm.generate([ids], SamplingParams(clover_mode=a.mode, clover_width=a.width,
            temperature=0, max_tokens=64))[0]
        torch.cuda.synchronize()
        wall = time.perf_counter() - begin
        after = keys()
        assert answer['token_ids'] and 'error' not in answer
        if a.warmup:
            assert before == after, 'Declared width captured a graph during the request'
        times = [v - begin for v in arrivals]
        decode_seconds = times[-1] - times[0]
        result['runs'].append(dict(repeat=index, **answer, wall_seconds=wall,
            ttft_seconds=times[0], end_to_end_tokens_per_second=len(answer['token_ids']) / wall,
            decode_tokens_per_second=((len(answer['token_ids']) - 1) / decode_seconds
                if decode_seconds > 0 else None),
            arrival_seconds=times, keys_before=before, keys_after=after))
        llm.cache_manager.check_integrity()
        save()
    assert all(r['token_ids'] == result['runs'][0]['token_ids'] for r in result['runs'])
    if a.audit:
        llm.rebuild_cache(moe_cache_size=cache.cache_size)
        result['rebuild_warmup'] = runtime.graph_warmup
        answer = llm.generate([ids], SamplingParams(clover_mode=a.mode, clover_width=a.width,
            temperature=0, max_tokens=64))[0]
        assert answer['token_ids'] == result['runs'][0]['token_ids']
        result['after_rebuild'] = answer
    result['state'] = 'passed'
    save()
finally:
    warmup.warmup_block_graphs = original
    llm.engine.shutdown()
