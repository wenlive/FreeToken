"""Use the shared ABAB metric logic with explicitly declared changing controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy

p = argparse.ArgumentParser()
p.add_argument('--labels', nargs=4, required=True)
p.add_argument('--vary-setting', action='append', default=[])
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path('C:/workspace/runs/T-20261104-clover-improve')
turns, inputs = [], []
for label in a.labels:
    paths = [p for p in sorted(root.glob('*-' + label + '/manifest.json'))
        if json.loads(p.read_text())['state'] == 'passed']
    assert len(paths) == 1, (label, paths)
    path = paths[0].parent / 'data/results.json'
    turns.append(json.loads(path.read_bytes()))
    inputs.append(dict(label=label, run=path.parent.parent.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), manifest=json.loads(paths[0].read_bytes())))
for left, right in ((0, 2), (1, 3)):
    assert turns[left]['settings'] == turns[right]['settings'], 'Repeated side changed its settings'
normalized = copy.deepcopy(turns)
for turn in normalized:
    for key in a.vary_setting:
        assert key not in ('warmups', 'repeats')
        del turn['settings'][key]
summary = runpy.run_path('benchmarks/clover/summarize_clover_abab.py')['summarize'](normalized)
summary.update(inputs=inputs, varied_settings={key: [turn['settings'][key] for turn in turns] for key in a.vary_setting},
    original_settings=[turn['settings'] for turn in turns])
(a.output / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(dict(varied_settings=summary['varied_settings'], cases=[dict(
    prompt=case['prompt'], mode=case['mode'], width=case['width'], equal=case['all_measured_output_ids_equal'],
    changes_percent={name: 100*(v['after_median']/v['before_median']-1) for name,v in case['metrics'].items() if v['before_median']}
) for case in summary['cases']]), indent=2))
