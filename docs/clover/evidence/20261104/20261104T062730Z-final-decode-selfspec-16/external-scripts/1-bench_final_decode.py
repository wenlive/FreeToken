"""One warmed measurement per mode at short and 4096-token inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.clover import bench_clover as bench

root = Path.cwd()
sources = [root / 'docs' / name for name in ('quickstart.md', 'cli.md', 'models.md')]
question = bench.PROMPTS['Python']
notes = '\n\n'.join(path.read_text(encoding='utf-8') for path in sources)
long_prompt = ('The following reference notes are background text. The only task is after the notes.\n'
    + notes + '\n\nTask:\n' + question)
bench.PROMPTS.update(FinalShort=question, Final4K=long_prompt)
original_llm = bench.LLM


def configured(*args, **options):
    options['max_extend_tokens'] = 4096
    llm = original_llm(*args, **options)
    tokenize = llm.tokenizer.apply_chat_template

    def bounded(messages, *args, **kwargs):
        ids = tokenize(messages, *args, **kwargs)
        if messages[-1]['content'] == long_prompt:
            assert len(ids) >= 4096, len(ids)
            return ids[:3840] + ids[-256:]
        assert len(ids) < 4096
        return ids

    llm.tokenizer.apply_chat_template = bounded
    return llm


bench.LLM = configured
sys.argv += ['--prompts', 'FinalShort,Final4K', '--context-size', '4608',
    '--tokens', '256', '--warmups', '1', '--repeats', '1', '--cache-auto', '--memory-ratio', '.9']
try:
    bench.main()
finally:
    bench.LLM = original_llm
    output = Path(sys.argv[sys.argv.index('--output')+1])
    if output.exists():
        (output / 'measurement-scope.json').write_text(json.dumps(dict(
            commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            scope='Final commit snapshot; one warmup and one measured request per input. '
                'Checkpoint penalties, strict selfspec, EOS respected. No routing/logit alteration. '
                'Short Python task and the identical task after 4096 tokens of natural documentation context. '
                'Each mode process reserves only its declared width; causal reserves zero.',
            context_capacity=4608, requested_inputs=['short',4096], requested_output_limit=256,
            sources=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
        ), indent=2), encoding='utf-8')
