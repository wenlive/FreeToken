"""Measure stock draft-self with the separately linked, algorithm-free probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--load-mode', choices=['mmap', 'none'], default='mmap')
p.add_argument('--prompts-json', type=Path, default=Path('C:/workspace/tasks/T-20261031-clover-gguf/llama-prompts/manifest.json'))
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
exe = Path('C:/workspace/builds/T-20261102-clover-q8/stock-preview-probe/clover-llama-preview.exe')
stock = Path('C:/workspace/builds/T-20261031-clover-gguf/llama-official-cb09-sm120/bin')
model = 'C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf'
prompts = json.loads(args.prompts_json.read_text(encoding='utf-8'))['prompts']
env = dict(os.environ)
env['PATH'] = str(stock) + ';' + env['PATH']
result = dict(state='running', stock_commit='cb09d4355d8bf968bb907c5de03cf42d8aa073ce',
    executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), cache_mib=8160, load_mode=args.load_mode,
    scope='one resident stock context per prompt/width; identical clean prefix; llama_perf decode and synchronized wall; 2 warmups + 5 measurements', cases=[])
def save():
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
save()
for name, prompt in prompts.items():
    for width in (8, 16):
        directory = args.output / f'{name}-W{width}'
        directory.mkdir()
        prompt_file = directory / 'prompt.txt'
        prompt_file.write_bytes((prompt['text'] + '\n').encode('utf-8'))
        command = [str(exe), '-m', model, '-f', str(prompt_file), '-c', '4096', '-b', '256', '-ub', '256',
            '-ngl', '99', '--cpu-moe', '--moe-cache-mib', '8160', '--load-mode', args.load_mode, '-t', '8', '-tb', '8',
            '-ctk', 'bf16', '-ctv', 'bf16', '--flash-attn', 'on', '--temp', '0', '--top-k', '1', '--seed', '42',
            '--repeat-penalty', '1', '--presence-penalty', '0', '--frequency-penalty', '0',
            '--spec-type', 'draft-self', '--spec-draft-n-max', str(width - 1)]
        (directory / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            subprocess.run(command, env=env, stdout=out, stderr=err, check=True, timeout=300)
        rows = []
        for line in (directory / 'stdout.log').read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('type') == 'input':
                assert row['ids'] == prompt['prompt_ids'], (name, 'prompt token mismatch')
            elif row.get('type') == 'preview':
                rows.append(row)
        assert len(rows) == 7 and all(r['decode_ms'] > 0 for r in rows)
        measured = [r for r in rows if not r['warmup']]
        result['cases'].append(dict(prompt=name, width=width, rows=rows,
            median_decode_ms=statistics.median(r['decode_ms'] for r in measured),
            median_wall_ms=statistics.median(r['wall_ms'] for r in measured), prompt_ids_verified=True))
        save()
        print(name, width, result['cases'][-1]['median_decode_ms'], flush=True)
result['state'] = 'complete'
save()
