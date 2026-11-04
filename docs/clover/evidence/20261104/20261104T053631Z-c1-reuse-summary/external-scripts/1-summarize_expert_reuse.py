"""Summarize measured cache reuse without treating snapshots as a policy benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
raw = a.input.read_bytes()
trace = json.loads(raw)
result = dict(input=str(a.input), sha256=hashlib.sha256(raw).hexdigest(), blocks=[],
    scope='Actual routes and cache snapshots. Excludes graph capture forwards from miss totals. '
    'A missing first-preview expert only matters if used again. Snapshot intersections cannot '
    'predict misses caused within a forward or the opportunity cost of protected slots. No speed claim.')
for block in trace['blocks']:
    forwards = [r for r in trace['forwards'] if r['block'] == block['block']]
    previews = [r for r in forwards if r['phase'] == 'preview']
    first = set(previews[0]['active_ids'])
    stable = [r for r in previews if not r['captured']]
    needed = [len(first.intersection(r['active_ids']).difference(r['resident_before']))
        for r in previews[1:] if not r['captured']]
    result['blocks'].append(dict(block=block['block'], width=previews[0]['rows'],
        first_union=len(first), cumulative_union=block['cumulative_union'], slots=block['slots'],
        preview_count=len(previews), noncapture_previews=len(stable),
        active_expert_calls=sum(layer[0] for r in stable for layer in r['stats_delta']),
        missed_expert_calls=sum(layer[1] for r in stable for layer in r['stats_delta']),
        first_missing_start=block['missing_first_at_next_start'],
        first_missing_and_reused_start=needed, verify_union=block['verify_union']))
result['by_width'] = {}
for width in sorted({r['width'] for r in result['blocks']}):
    rows = [r for r in result['blocks'] if r['width'] == width]
    active, missing = [sum(r[key] for r in rows) for key in ('active_expert_calls', 'missed_expert_calls')]
    result['by_width'][str(width)] = dict(blocks=len(rows),
        first_union_range=[min(r['first_union'] for r in rows), max(r['first_union'] for r in rows)],
        cumulative_union_range=[min(r['cumulative_union'] for r in rows), max(r['cumulative_union'] for r in rows)],
        blocks_with_first_eviction=sum(any(r['first_missing_start']) for r in rows),
        blocks_with_first_evicted_and_reused=sum(any(r['first_missing_and_reused_start']) for r in rows),
        first_missing_and_reused=sum(sum(r['first_missing_and_reused_start']) for r in rows),
        active_expert_calls=active, missed_expert_calls=missing, miss_fraction=missing/max(1,active))
result['state'] = 'complete'
(a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result['by_width'], indent=2))
