"""Compact the prefill localization evidence without turning reporting levels into tolerances."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
runs=Path('C:/workspace/runs/T-20261104-clover-improve')
result=dict(state='running',inputs=[],cases=[],scope='Five short prompts, per-layer same-Q8 HF, native tile '
    'perturbation and sparse admission disabled. GDN oracle receives identical real operands. '
    'Diagnostics do not prove all arithmetic differences harmless; JSON W32 progress remains unresolved.')

def read(label):
    matches=[m for m in runs.glob('*-'+label+'/manifest.json') if json.loads(m.read_bytes())['state']=='passed']
    assert len(matches)==1,(label,matches)
    path=matches[0].parent/'data/results.json'
    data=json.loads(path.read_bytes())
    assert data['state']=='complete'
    result['inputs'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        manifest=json.loads(matches[0].read_bytes())))
    return data

native=read('b1-prefill-none-sparse128')
comparison=read('b1-prefill-hf-comparison')
for case in comparison['cases']:
    name=case['prompt']
    hf=case['against_hf'][0]['comparison']
    tile,sparse=[row['comparison'] for row in case['native_controls']]
    operators=[row for row in native['gdn_operator_cases'] if row['prompt']==name]
    assert len(operators)==30 and len(hf['layers'])==len(tile['layers'])==len(sparse['layers'])==40
    exact_sparse=all(error['max_abs']==0 and error['finite']
        for layer in sparse['layers'] for error in layer['errors'].values())
    result['cases'].append(dict(prompt=name,first_layer=hf['layers'][0],
        hf_first_exceeding_diagnostic_levels=hf['first_layer_exceeding_diagnostic_levels'],
        final_hf_layer=hf['layers'][-1],final_tile_layer=tile['layers'][-1],
        sparse_all_layer_tensors_exact=exact_sparse,
        gdn_same_operand_max={metric:max(row[metric]['relative_l2'] for row in operators)
            for metric in ('output_vs_hf','state_vs_hf')},
        gdn_first_layer=next(row for row in operators if row['layer']==0)))
result.update(state='complete',all_sparse_layer_tensors_exact=all(c['sparse_all_layer_tensors_exact'] for c in result['cases']))
(a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(state=result['state'],cases=len(result['cases']),all_sparse_layer_tensors_exact=result['all_sparse_layer_tensors_exact'])))
