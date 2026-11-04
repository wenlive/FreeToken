"""Retain every A3 process turn; separate initial compilation from graph capture."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path('C:/workspace/runs/T-20261104-clover-improve')
summary = dict(state='measured', full_compatibility='pending separate gate', cases=[],
    protocol='One uninterrupted serial queue, ABAB processes per mode; three requests per process. '
    'Expert cache reset and prefix reuse disabled for each request. Startup and capture are separate from arrival timing. '
    'The initial selfspec A turn includes first-use compilation and is retained but not used as a capture-only speedup claim.')
for mode in ('selfspec', 'denoise'):
    turns = []
    all_ids = []
    for index, label in enumerate(('a', 'b', 'a', 'b'), 1):
        candidates = list(root.glob(f'*-a3-{mode}-abab-{index}-{label}/data/results.json'))
        assert len(candidates) == 1, candidates
        path = candidates[0]
        raw = path.read_bytes()
        data = json.loads(raw)
        assert data['state'] == 'passed'
        all_ids.extend(row['token_ids'] for row in data['runs'])
        turns.append(dict(variant=label, run=path.parent.parent.name,
            sha256=hashlib.sha256(raw).hexdigest(), startup_seconds=data['startup_seconds'],
            warmup=data['warmup'], cache_slots=data['cache_slots'],
            first={key: data['runs'][0][key] for key in ('wall_seconds', 'ttft_seconds',
                'decode_tokens_per_second', 'end_to_end_tokens_per_second')},
            warm_wall_seconds=[row['wall_seconds'] for row in data['runs'][1:]],
            capture_during_request=data['runs'][0]['keys_before'] != data['runs'][0]['keys_after']))
    assert all(ids == all_ids[0] for ids in all_ids)
    summary['cases'].append(dict(mode=mode, all_output_ids_equal=True, turns=turns,
        first_request_after_median_seconds=statistics.median(t['first']['wall_seconds'] for t in turns if t['variant']=='b'),
        hot_request_after_median_seconds=statistics.median(v for t in turns if t['variant']=='b' for v in t['warm_wall_seconds'])))
integrity = list(root.glob('*-a3-q8-integrity/data/results.json'))
assert len(integrity) == 1
data = json.loads(integrity[0].read_bytes())
assert data['state'] == 'passed' and len(data['audits']) == 2
assert all(row['before'] == row['after'] for row in data['audits'])
summary['integrity'] = dict(run=integrity[0].parent.parent.name, startup_and_rebuild_identical=True,
    sha256=hashlib.sha256(integrity[0].read_bytes()).hexdigest())
(a.output / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
