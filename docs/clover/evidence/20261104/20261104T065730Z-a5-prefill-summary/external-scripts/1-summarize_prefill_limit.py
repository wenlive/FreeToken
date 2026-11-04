"""Summarize recorded sparse-prefill ABAB without treating one-token latency as decode."""
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
result=dict(state='running',inputs=[],cases=[],scope='Same-boot ABAB, cold expert cache and no prefix reuse. '
    'Each formal request emits one token, so decode TPS is undefined. Forward CUDA-event diagnostics '
    'are separate from primary latency samples. Identical input/output checks do not prove full-generation equivalence.')
for slots in (2292,2928):
    turns=[]
    for turn,limit in enumerate((128,1024,128,1024),1):
        paths=[path for path in root.glob(f'*-a5-prefill-{slots}-abab-{turn}/manifest.json')
            if json.loads(path.read_bytes())['state']=='passed']
        assert len(paths)==1,paths
        path=paths[0].parent/'data/results.json'
        data=json.loads(path.read_bytes())
        manifest=json.loads(paths[0].read_bytes())
        assert data['state']=='complete' and not manifest['dirty']
        assert data['settings']['cache_slots']==slots and data['settings']['limit']==limit
        result['inputs'].append(dict(slots=slots,turn=turn,limit=limit,source=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),manifest=manifest))
        turns.append(data)
    stable=lambda data:{k:v for k,v in data['settings'].items() if k not in ('output','limit')}
    assert all(stable(data)==stable(turns[0]) and data['inputs']==turns[0]['inputs']
        and data['corpus_sha256']==turns[0]['corpus_sha256'] for data in turns)
    for item in turns[0]['inputs']:
        length=item['length']
        samples=[[row for row in data['runs'] if row['length']==length and not row['warmup']] for data in turns]
        assert all(len(rows)==5 for rows in samples)
        assert all(row['decode_tokens_per_second'] is None for rows in samples for row in rows)
        signatures=[sorted({tuple(row['token_ids']) for row in rows}) for rows in samples]
        metrics={}
        for metric in ('ttft_seconds','wall_seconds','end_to_end_tokens_per_second'):
            values=[[row[metric] for row in rows] for rows in samples]
            medians=[statistics.median(values) for values in values]
            metrics[metric]=dict(process_medians=medians,process_ranges=[[min(v),max(v)] for v in values],
                dense_median=statistics.median(values[0]+values[2]),
                sparse_median=statistics.median(values[1]+values[3]),
                sparse_over_dense_pairs=[medians[i+1]/medians[i] for i in (0,2)])
        forwards=[next(row for row in data['forwards'] if row['length']==length) for data in turns]
        result['cases'].append(dict(cache_slots=slots,input_tokens=length,
            all_formal_output_ids_equal=len({ids for values in signatures for ids in values})==1,
            output_signatures_per_turn=signatures,metrics=metrics,
            separate_forward_milliseconds=[row['milliseconds'] for row in forwards],
            forward_output_ids=[row['token_ids'] for row in forwards]))
result['state']='complete'
(a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(state=result['state'],cases=[dict(cache_slots=row['cache_slots'],input_tokens=row['input_tokens'],
    equal=row['all_formal_output_ids_equal'],ttft=row['metrics']['ttft_seconds']) for row in result['cases']]),indent=2))
