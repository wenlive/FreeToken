"""Separate forward-event envelopes for the already measured embedding placement comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
root=Path('C:/workspace/runs/T-20261104-clover-improve')
result=dict(state='running',sources=[],cases=[],scope='Separate instrumented diagnostics matching '
    'the prior uninstrumented C2 cap256/W0-causal and cap256/W32-six-mode settings. '
    'Per-request mean forward-event costs, then median across two formal requests. '
    'Round envelopes contain nested preview/verify/state events: do not add them together. '
    'No ABAB or end-to-end speedup claim is derived from these two diagnostic processes.')
for maximum in (0,32):
    variants={}
    for placement in ('device','host'):
        label=f'c2-forward-w{maximum}-{placement}'
        paths=[path for path in root.glob('*-'+label+'/manifest.json')
            if json.loads(path.read_bytes())['state']=='passed']
        assert len(paths)==1,paths
        manifest=json.loads(paths[0].read_bytes())
        path=paths[0].parent/'data/results.json'
        data=json.loads(path.read_bytes())
        assert data['state']=='complete' and not manifest['dirty']
        assert data['settings']['embedding_host']==(placement=='host')
        assert data['settings']['token_budget']==256 and data['settings']['repeats']==2
        result['sources'].append(dict(label=label,result=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),manifest=manifest))
        variants[placement]=data
    stable=lambda data:{k:v for k,v in data['settings'].items() if k not in ('embedding_host','mapped_host_table_bytes','cache_slots')}
    assert stable(variants['device'])==stable(variants['host'])
    assert variants['device']['prompts']==variants['host']['prompts']
    cases=sorted({(r['prompt'],r['mode'],r['width']) for r in variants['device']['runs'] if not r['warmup']})
    for key in cases:
        values={}
        all_ids=[]
        for placement,data in variants.items():
            rows=[r for r in data['runs'] if not r['warmup'] and (r['prompt'],r['mode'],r['width'])==key]
            assert len(rows)==2
            all_ids += [r['token_ids'] for r in rows]
            phases={}
            for row in rows:
                phases_in_row={event['phase'] for event in row['runtime_breakdown']}
                for phase in phases_in_row:
                    timings=[event['cuda_ms'] for event in row['runtime_breakdown'] if event['phase']==phase]
                    phases.setdefault(phase,[]).append(dict(calls=len(timings),mean_ms=statistics.mean(timings),total_ms=sum(timings)))
            values[placement]=dict(cache_slots=data['settings']['cache_slots'],
                phases={phase:dict(per_request=records,median_mean_ms=statistics.median(r['mean_ms'] for r in records))
                    for phase,records in phases.items()})
        result['cases'].append(dict(maximum=maximum,prompt=key[0],mode=key[1],width=key[2],
            all_formal_output_ids_equal=all(ids==all_ids[0] for ids in all_ids),variants=values))
result['state']='complete'
(a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(cases=len(result['cases']),output_differences=[{k:v for k,v in r.items() if k!='variants'}
    for r in result['cases'] if not r['all_formal_output_ids_equal']],python=[r for r in result['cases'] if r['prompt']=='Python']),indent=2))
