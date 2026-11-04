"""Reduce completed fixed-history comparisons while retaining per-row acceptance observations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
runs=Path('C:/workspace/runs/T-20261104-clover-improve')
paths=[m for m in runs.glob('*-b0-three-way-compare/manifest.json') if json.loads(m.read_bytes())['state']=='passed']
assert len(paths)==1
path=paths[0].parent/'data/results.json'
data=json.loads(path.read_bytes())
assert data['state']=='complete'
result=dict(state='complete',source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    manifest=json.loads(paths[0].read_bytes()),cases=[],groups=[],
    scope='Same fixed histories and preview blocks; HF semantic reference, stock llama numerical ruler. '
    'Per-row 1.2*max(llama-HF,native-noise) observations are preserved, not a replacement quality criterion. '
    'Causal and preview groups are separate; all previews are single forwards, not a LowConfidence benchmark.')
groups={}
for case in data['cases']:
    for part in case['comparisons']:
        metrics={name:{'mean_l2':statistics.mean(values['relative_l2_per_row']),
                       'max_l2':max(values['relative_l2_per_row']),
                       'mean_kl':statistics.mean(values['kl_per_row']),
                       'max_kl':max(values['kl_per_row']),
                       'top1_agreement':statistics.mean(values['top1_equal_per_row'])}
                 for name,values in part['metrics'].items()}
        envelopes={name:dict(rows=len(value['within']),within=sum(value['within']),
                            outlier_rows=[i for i,ok in enumerate(value['within']) if not ok])
                   for name,value in part['heuristic_envelope'].items()}
        result['cases'].append(dict(prompt=case['prompt'],part=part['part'],metrics=metrics,envelopes=envelopes))
        groups.setdefault(part['part'],[]).append(part)
for name,parts in groups.items():
    summary={}
    for metric in ('relative_l2_per_row','kl_per_row'):
        flags=[ok for part in parts for ok in part['heuristic_envelope'][metric]['within']]
        summary[metric]=dict(rows=len(flags),within=sum(flags),fraction=sum(flags)/len(flags))
    result['groups'].append(dict(part=name,heuristic_envelope=summary,
        mean_top1={pair:statistics.mean([ok for part in parts for ok in part['metrics'][pair]['top1_equal_per_row']])
                  for pair in ('native_hf','llama_hf','native_noise','native_llama')}))
(a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result['groups'],indent=2))
