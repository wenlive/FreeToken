"""Report progress differences against measured native perturbations, without changing tolerances."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--native-prefix', required=True)
p.add_argument('--reference-label', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
runs = Path('C:/workspace/runs/T-20261104-clover-improve')
result = dict(state='running', cases=[], inputs=[],
    scope='Four native arithmetic variants and unchanged HF LowConfidence. Full-generation '
    'histories can diverge; report first-block progress separately. Native noise envelopes are '
    'observations from these perturbations, not proofs of correctness or a revised tolerance.')


def read(label):
    candidates = [p for p in sorted(runs.glob('*-' + label + '/manifest.json'))
        if json.loads(p.read_text())['state'] == 'passed']
    assert len(candidates) == 1, (label,candidates)
    path = candidates[0].parent / 'data/results.json'
    data = json.loads(path.read_bytes())
    assert data['state'] == 'complete', (label,data['state'])
    result['inputs'].append(dict(label=label, result=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), manifest=json.loads(candidates[0].read_bytes())))
    return data


native = {phase: read(a.native_prefix + phase) for phase in ('none','prefill','preview','both')}
reference = read(a.reference_label)
baseline = native['none']
for data in native.values():
    assert data['settings'] == baseline['settings'] and data['prompts'] == baseline['prompts']


def progress(blocks):
    assert blocks and all(steps for steps in blocks)
    return dict(tokens_per_preview=sum(map(sum,blocks))/sum(map(len,blocks)),
        first_block_tokens_per_preview=sum(blocks[0])/len(blocks[0]),
        preview_calls=sum(map(len,blocks)), block_steps=blocks)


for hf in reference['cases']:
    name,width = hf['prompt'],hf['width']
    assert baseline['prompts'][name]['token_ids'] == hf['prompt_ids']
    values,outputs = {},{}
    for phase,data in native.items():
        cases = [r for r in data['runs'] if not r['warmup'] and r['prompt']==name
            and r['mode']=='denoise' and r['width']==width]
        assert len(cases)==1
        row=cases[0]
        values[phase]=progress([r['steps'] for r in row['rounds']])
        outputs[phase]=row['token_ids']
    values['hf']=progress(hf['blocks'])
    envelopes={}
    for metric in ('tokens_per_preview','first_block_tokens_per_preview'):
        observed=[values[phase][metric] for phase in native]
        center=values['none'][metric]
        radius=max(abs(value-center) for value in observed)
        difference=abs(values['hf'][metric]-center)
        envelopes[metric]=dict(native_range=[min(observed),max(observed)],
            native_noise_radius=radius,hf_absolute_difference=difference,
            hf_within_observed_range=min(observed)<=values['hf'][metric]<=max(observed),
            hf_within_1_2_noise_radius=difference<=1.2*radius,
            hf_relative_difference=values['hf'][metric]/center-1)
    row=dict(prompt=name,width=width,progress=values,envelopes=envelopes,
        output_equal_to_baseline={phase:ids==outputs['none'] for phase,ids in outputs.items()},
        hf_output_equal=hf['visible_token_ids']==outputs['none'])
    result['cases'].append(row)
    print(json.dumps(dict(prompt=name,width=width,envelopes=envelopes),ensure_ascii=False))
result['state']='complete'
(a.output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
