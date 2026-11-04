"""Summarize natural-context progress and answer checks without a speedup claim."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
runs=Path('C:/workspace/runs/T-20261104-clover-improve')
result=dict(state='running',inputs=[],cases=[],scope='Two tasks per context, one instrumented request per variant. '
    'Final numeric marker checks only, not a general quality benchmark. No warmup; capture cost may affect '
    'timings, so these are progress/quality diagnostics, not ABAB performance evidence. Histories can diverge.')

def read(prefix,variant):
    matches=[m for m in runs.glob(f'*-b3-natural-{prefix}-{variant}/manifest.json')
        if json.loads(m.read_bytes())['state']=='passed']
    assert len(matches)==1
    path=matches[0].parent/'data/results.json'
    data=json.loads(path.read_bytes())
    assert data['state']=='complete'
    result['inputs'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        manifest=json.loads(matches[0].read_bytes())))
    return data

for prefix in (8192,16384,32768):
    data={v:read(prefix,v) for v in ('default','noise','split8')}
    base=data['default']
    for value in data.values():
        assert value['settings']==base['settings']
        assert value['prompts']==base['prompts']
    for name,expected in [('NaturalA',42),('NaturalB',52)]:
        assert len(base['prompts'][name]['token_ids'])==prefix
        observations={}
        ids={}
        for variant,value in data.items():
            rows=[r for r in value['runs'] if not r['warmup'] and r['prompt']==name]
            assert len(rows)==1
            row=rows[0]
            assert row['mode']=='denoise' and row['width']==16
            steps=[block['steps'] for block in row['rounds']]
            progress=sum(map(sum,steps))/sum(map(len,steps))
            assert abs(progress-row['tokens_per_preview'])<1e-10
            answers=re.findall(r'####\s*([+-]?\d+)',row['text'])
            ids[variant]=row['token_ids']
            observations[variant]=dict(text=row['text'],output_tokens=row['output_tokens'],
                reached_cap=row['output_tokens']>=value['settings']['token_budget'],
                final_marker=int(answers[-1]) if answers else None,
                final_marker_correct=bool(answers) and int(answers[-1])==expected,
                tokens_per_preview=progress,visible_tokens_per_preview=row['visible_tokens_per_preview'],
                first_block_tokens_per_preview=sum(steps[0])/len(steps[0]),steps=steps)
        envelopes={}
        for metric in ('tokens_per_preview','first_block_tokens_per_preview'):
            center=observations['default'][metric]
            noise=abs(observations['noise'][metric]-center)
            split=abs(observations['split8'][metric]-center)
            envelopes[metric]=dict(noise_absolute_difference=noise,split_absolute_difference=split,
                split_within_1_2_single_noise=split<=noise*1.2)
        result['cases'].append(dict(prefix_tokens=prefix,prompt=name,expected=expected,
            observations=observations,envelopes=envelopes,
            output_equal_to_default={v:t==ids['default'] for v,t in ids.items()}))
result.update(state='complete',all_18_final_markers_correct=all(
    o['final_marker_correct'] for c in result['cases'] for o in c['observations'].values()),
    any_output_capped=any(o['reached_cap'] for c in result['cases'] for o in c['observations'].values()))
(a.output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('state','all_18_final_markers_correct','any_output_capped')}))
