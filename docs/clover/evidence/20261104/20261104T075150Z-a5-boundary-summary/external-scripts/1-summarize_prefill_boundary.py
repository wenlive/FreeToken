"""Bind existing candidate-cutoff outputs to their exact sources and settings."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
root=Path('C:/workspace/runs/T-20261104-clover-improve')
result=dict(state='running',sources=[],cases=[],scope='Complete generated-ID equality for '
    'the recorded boundary fixtures; does not establish all-input bitwise equivalence.')
for slots in (2292,2928):
    turns=[]
    for limit in (128,256):
        paths=[path for path in root.glob(f'*-a5-boundary-{slots}-limit{limit}/manifest.json')
            if json.loads(path.read_bytes())['state']=='passed']
        assert len(paths)==1,paths
        manifest=json.loads(paths[0].read_bytes())
        path=paths[0].parent/'data/results.json'
        data=json.loads(path.read_bytes())
        assert data['state']=='complete' and not manifest['dirty']
        assert data['settings']['cache_slots']==slots and data['settings']['limit']==limit
        result['sources'].append(dict(manifest=manifest,result=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        turns.append(data)
    left,right=turns
    assert {k:v for k,v in left['settings'].items() if k!='limit'}=={k:v for k,v in right['settings'].items() if k!='limit'}
    assert left['corpus_sha256']==right['corpus_sha256']
    assert len(left['cases'])==len(right['cases'])==12
    for length in (128,129,192,256,257,384):
        values=[[row for row in data['cases'] if row['input_length']==length] for data in turns]
        assert all(len(rows)==2 and [row['repeat'] for row in rows]==[0,1] for rows in values)
        flat=[row for rows in values for row in rows]
        assert all(row['prompt_ids']==flat[0]['prompt_ids'] for row in flat)
        outputs=[row['token_ids'] for row in flat]
        result['cases'].append(dict(cache_slots=slots,input_length=length,
            before_repeats_equal=outputs[0]==outputs[1],after_repeats_equal=outputs[2]==outputs[3],
            all_outputs_equal=all(ids==outputs[0] for ids in outputs),
            output_lengths=list(map(len,outputs)),
            first_difference=next((i for i,(x,y) in enumerate(zip(outputs[0],outputs[2])) if x!=y),
                min(len(outputs[0]),len(outputs[2])) if len(outputs[0])!=len(outputs[2]) else None)))
result['state']='passed' if all(row['all_outputs_equal'] for row in result['cases']) else 'output_difference'
(a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(state=result['state'],cases=result['cases']),indent=2))
# A diagnostic difference is preserved for review. It must not automatically enable a new default.
