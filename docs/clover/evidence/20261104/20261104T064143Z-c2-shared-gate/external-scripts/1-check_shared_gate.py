"""Compare candidate outputs with main and bind graph/lifecycle/regression gates."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--phase', choices=['a3', 'd2', 'c2', 'ports'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path('C:/workspace/runs/T-20261104-clover-improve')
candidate = 'nvfp4' if a.phase == 'ports' else 'q8'
result = dict(state='running', inputs=[], comparisons=[], gates={})

def read(label):
    paths = [p for p in sorted(root.glob('*-' + label + '/manifest.json'))
        if json.loads(p.read_text())['state'] == 'passed']
    assert paths, label
    manifest_path = paths[-1]
    manifest = json.loads(manifest_path.read_bytes())
    path = manifest_path.parent / 'data/results.json'
    data = json.loads(path.read_bytes())
    result['inputs'].append(dict(label=label, run=manifest_path.parent.name,
        source=manifest['commit'], result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()))
    return data

for mode in ('off', 'on'):
    before = read('phase1-mtp-main-' + mode)
    after = read(f'{a.phase}-mtp-{candidate}-{mode}')
    for x, y in zip(before['cases'], after['cases'], strict=True):
        assert x['prompt'] == y['prompt'] and x['prompt_ids'] == y['prompt_ids']
        result['comparisons'].append(dict(mode='mtp-' + mode, prompt=x['prompt'],
            equal=x['token_ids'] == y['token_ids'], tokens=len(x['token_ids'])))
before, after = read('phase1-qwen36-main'), read(f'{a.phase}-qwen36-{candidate}')
for x, y in zip(before['cases'], after['cases'], strict=True):
    assert x['prompt'] == y['prompt'] and x['prompt_ids'] == y['prompt_ids']
    result['comparisons'].append(dict(mode='qwen36', prompt=x['prompt'],
        equal=x['token_ids'] == y['token_ids'], tokens=len(x['token_ids'])))
for gate in ('112-tails', 'lifecycle', 'regression'):
    result['gates'][gate] = read(f'{a.phase}-{candidate}-{gate}')
result['state'] = 'passed' if all(r['equal'] for r in result['comparisons']) else 'failed'
(a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(state=result['state'], comparisons=result['comparisons'],
    regression=result['gates']['regression']['summary']), indent=2))
assert result['state'] == 'passed'
