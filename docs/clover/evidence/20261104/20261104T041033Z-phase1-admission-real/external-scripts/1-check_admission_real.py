"""A rejected Clover request owns no pages or state; subsequent requests still run."""
import argparse
import json
from pathlib import Path

from freetoken.core import SamplingParams
from freetoken.llm import LLM

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
llm = LLM(a.model, moe_strategy='offload', moe_cache_auto=True, memory_ratio=.9,
    kv_reserve_tokens=4096, num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, max_running_req=1, clover_max_width=32,
    expert_load='serial', cuda_graph_bs=[1])
rows = []

def capacity():
    return dict(pages=llm.cache_manager.available_size,
        states=llm.engine.linear_state_pool.num_free_slots,
        tables=llm.table_manager.available_size)

try:
    cases = [([64] * 4090, SamplingParams(clover_mode='denoise', max_tokens=1)),
        ([64] * 4081, SamplingParams(clover_mode='denoise', max_tokens=10)),
        ([64] * 8, SamplingParams(clover_mode='selfspec', clover_width=32, max_tokens=16)),
        ([64] * 8, SamplingParams(clover_mode='denoise', temperature=1, max_tokens=16))]
    for tokens, params in cases:
        before = capacity()
        result = llm.generate([tokens], params)[0]
        after = capacity()
        assert result.get('error') and not result['token_ids'], result
        assert before == after, (before, after)
        llm.cache_manager.check_integrity()
        rows.append(dict(prompt_tokens=len(tokens), parameters=vars(params),
            result=result, before=before, after=after))
    prompt = llm.tokenizer.apply_chat_template([dict(role='user', content='Count from 1 to 10.')],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    good = llm.generate([prompt], SamplingParams(clover_mode='causal', max_tokens=32))[0]
    assert good['token_ids'] and 'error' not in good, good
    llm.cache_manager.check_integrity()
    (a.output / 'results.json').write_text(json.dumps(dict(state='passed', rejected=rows,
        subsequent_valid=good), indent=2), encoding='utf-8')
finally:
    llm.engine.shutdown()
