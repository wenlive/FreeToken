"""Independently regenerate canonical prompts from the pinned official HF tokenizer."""
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer
sys.path.insert(0, 'C:/workspace/wt/T-20261102-clover-q8/FreeToken/benchmarks')
from bench_clover import PROMPTS

source = 'C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff'
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
root = Path('C:/workspace/runs/T-20261102-clover-q8')
actual = json.loads((root / 'official-q8-p0-full-matrix/results.json').read_text(encoding='utf-8'))
recorded = json.loads(Path('C:/workspace/tasks/T-20261031-clover-gguf/llama-prompts/manifest.json').read_text(encoding='utf-8'))
out = Path(__file__).parent / 'short-prompts'
out.mkdir(exist_ok=False)
rows = {}
for name, user in PROMPTS.items():
    text = tokenizer.apply_chat_template([dict(role='user', content=user)], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert ids == actual['prompts'][name]['token_ids'] == recorded['prompts'][name]['prompt_ids']
    assert text == recorded['prompts'][name]['text']
    path = out / (name + '.txt')
    path.write_bytes(text.encode('utf-8'))
    rows[name] = dict(text=text, user_text=user, prompt_ids=ids, tokens=len(ids), file=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(out / 'manifest.json').write_text(json.dumps(dict(state='complete', source=source,
    source_revision='4b0402b65e523d0183a14c431955438714cb16ff',
    matches_actual_official_q8_native_ids=True, matches_recorded_stock_inputs=True, prompts=rows), indent=2), encoding='utf-8')
print({name: row['tokens'] for name, row in rows.items()})
