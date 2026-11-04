"""Long natural contexts: native/split and preview-tile noise, with progress traces."""
import argparse
import json
from pathlib import Path
import sys

from benchmarks.clover import bench_clover as bench
from freetoken.core import get_global_ctx
from freetoken.kernel.triton import attention

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--attention-variant', choices=['default', 'split8', 'noise'], required=True)
p.add_argument('--prefix-tokens', type=int, required=True)
p.add_argument('--corpus', type=Path, required=True)
a, rest = p.parse_known_args()
rows = [json.loads(line) for line in a.corpus.read_text(encoding='utf-8').splitlines()]
questions = [
    'A community garden has 6 rows with 7 tomato plants in each row. Explain how to calculate the total, '
    'give a brief check using repeated addition, and end with #### followed by the total.',
    '图书角原有 37 本书，又收到 15 本捐书。请用中文解释计算过程，给出一个检验方法，最后以 #### 加总数结尾。',
]
for label, offset, question in zip(('NaturalA', 'NaturalB'), (0, 400), questions):
    body = '\n\n'.join(row['question'] + '\n' + row['answer'] for row in rows[offset:offset+400])
    bench.PROMPTS[label] = 'Worked examples for reference:\n' + body + '\n\nNow solve only this new problem:\n' + question
original_llm = bench.LLM

def configured(*args, **options):
    options.update(max_extend_tokens=4096, clover_split_kv=a.attention_variant == 'split8')
    llm = original_llm(*args, **options)
    tokenize = llm.tokenizer.apply_chat_template

    def exact_prefix(*args, **kwargs):
        ids = tokenize(*args, **kwargs)
        assert len(ids) > a.prefix_tokens > 256
        # Retain the full final question and generation marker; trim only example text.
        return ids[:a.prefix_tokens-256] + ids[-256:]

    llm.tokenizer.apply_chat_template = exact_prefix
    return llm

select = attention._select_extend_tile

def noise(*args):
    m, n = select(*args)
    if getattr(get_global_ctx().batch, 'clover_preview', False):
        assert n > 16
        return m, n // 2
    return m, n

bench.LLM = configured
if a.attention_variant == 'noise':
    attention._select_extend_tile = noise
sys.argv = [sys.argv[0], *rest, '--prompts', 'NaturalA,NaturalB', '--context-size', str(a.prefix_tokens+256)]
try:
    bench.main()
finally:
    bench.LLM = original_llm
    attention._select_extend_tile = select
