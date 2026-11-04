"""Exercise public tool help and the isolated grading worker after relocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
commands = [[], ['bench'], ['context'], ['preview'], ['preview', 'graphs'],
    ['preview', 'lifecycle'], ['preview', 'verify'], ['preview', 'split-kv'],
    ['preview', 'outputs'], ['costs'], ['costs', 'abab'], ['hf-check'],
    ['hf-check', 'denoise'], ['hf-check', 'selfspec'], ['hf-check', 'quantization'],
    ['quality'], ['quality', 'fetch-data']]
if Path('benchmarks/clover/bench_clover_llama.py').exists():
    commands.extend([['llama-compare'], ['preview', 'ab']])
rows = []
for parts in commands:
    command = [sys.executable, '-m', 'benchmarks.clover', *parts, '--help']
    completed = subprocess.run(command, capture_output=True, text=True)
    rows.append(dict(command=command, code=completed.returncode,
        stdout=completed.stdout, stderr=completed.stderr))
    (a.output / 'results.json').write_text(json.dumps(dict(state='running', commands=rows), indent=2), encoding='utf-8')
    assert completed.returncode == 0, completed.stderr
from benchmarks.clover.eval_clover_quality import grade_code
import inspect
print(inspect.signature(grade_code))
(a.output / 'results.json').write_text(json.dumps(dict(state='passed', commands=rows), indent=2), encoding='utf-8')
