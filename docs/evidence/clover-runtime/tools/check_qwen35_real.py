"""Compare the same real Qwen3.5-family NVFP4 checkpoint on three source trees."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--code-root', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(args.code_root / 'python'))
import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.mm.config import MultimodalConfig

args.output.mkdir(parents=True, exist_ok=False)
assert Path(inspect.getfile(LLM)).resolve().is_relative_to(args.code_root.resolve())
torch.set_num_threads(4)
settings = dict(moe_strategy='offload', moe_cache_size=4096, expert_load='serial',
    max_running_req=1, memory_ratio=.85, num_token_override=4096,
    max_seq_len_override=4096, max_extend_tokens=1024, cuda_graph_bs=[1])
protected = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in args.model.iterdir()
    if f.is_file() and f.suffix == '.json'}
result = dict(state='running', model=str(args.model), settings=settings, disabled_encoders=['vision'],
    source_commit=subprocess.check_output(['git', '-C', str(args.code_root), 'rev-parse', 'HEAD'], text=True).strip(),
    protected_before=protected, cases=[])
def save():
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
save()
llm = LLM(str(args.model), **settings, mm=MultimodalConfig(disabled_encoders=frozenset({'vision'})))
try:
    for name, text in [('Count', 'Count from 1 to 10.'), ('Arithmetic', '计算17+25，只写结果。'),
        ('JSON', 'Return only a JSON object with the keys city and country for Paris.')]:
        ids = llm.tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        begin = time.perf_counter()
        answer = llm.generate([ids], SamplingParams(temperature=0, max_tokens=64))[0]
        result['cases'].append(dict(prompt=name, prompt_ids=ids, **answer, seconds=time.perf_counter() - begin))
        save()
    result['protected_after'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in args.model.iterdir()
        if f.is_file() and f.suffix == '.json'}
    assert result['protected_before'] == result['protected_after']
    result['state'] = 'complete'
    save()
finally:
    llm.engine.shutdown()
