"""Compare controller decisions on the exact saved native JSON first-block inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
queue=json.loads(Path(__file__).with_name('json-trajectory-queue.json').read_bytes())
assert queue['state']=='passed'
paths=dict(native=Path(queue['baseline']),**{k:Path(v) for k,v in queue['completed'].items()})
reports={key:json.loads(path.read_bytes()) for key,path in paths.items()}
native_case=reports['native']['cases'][0]
assert len(reports['native']['cases'])==1
data={}
for name in ('native','noise','hf'):
    case=reports[name]['cases'][0]
    artifact=case['artifact']
    path=paths[name].parent/artifact['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']
    data[name]=torch.load(path,map_location='cpu',weights_only=True)['previews']
llama=reports['llama']['cases'][0]
data['llama']=[]
for item in llama['files'][1:]:
    path=paths['llama'].parent/item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    data['llama'].append(torch.from_numpy(np.fromfile(path,dtype='<f4').reshape(item['rows'],-1)))
assert all(len(value)==len(native_case['trajectory']) for value in data.values())
rows=[]
acc={name:dict(top1_equal=0,pending_rows=0,decision_positions_equal=0,
              threshold_crossings=0,confidence_absolute_difference_sum=0.) for name in ('noise','hf','llama')}
dist={name:dict(kl=[],l2=[],top1=[]) for name in ('native_hf','llama_hf','native_noise')}
envelope={key:[] for key in ('relative_l2_per_row','kl_per_row')}
for step,trace in enumerate(native_case['trajectory']):
    pending=[i for i,token in enumerate(trace['input_ids']) if token==native_case['mask_id']]
    decisions={}
    for name,scores in data.items():
        logits=scores[step].clone().float()
        logits[:,native_case['mask_id']]=-float('inf')
        probabilities=logits.softmax(-1)
        confidence,picks=probabilities.max(-1)
        selected=[i for i in pending if float(confidence[i])>.95]
        fallback=not selected
        if fallback:
            selected=[max(pending,key=lambda i:float(confidence[i]))]
        decisions[name]=dict(selected=selected,selected_tokens=[int(picks[i]) for i in selected],
            count=len(selected),fallback=fallback,pending_predictions=picks[pending].tolist(),
            pending_confidences=confidence[pending].tolist(),max_pending_confidence=float(confidence[pending].max()))
    assert decisions['native']['selected']==trace['selected']
    assert decisions['native']['pending_predictions']==[trace['prediction'][i] for i in pending]
    for name in acc:
        actual,reference=decisions[name],decisions['native']
        acc[name]['decision_positions_equal']+=actual['selected']==reference['selected']
        acc[name]['pending_rows']+=len(pending)
        acc[name]['top1_equal']+=sum(a==b for a,b in zip(actual['pending_predictions'],reference['pending_predictions'],strict=True))
        acc[name]['threshold_crossings']+=sum((a>.95)!=(b>.95) for a,b in zip(actual['pending_confidences'],reference['pending_confidences'],strict=True))
        acc[name]['confidence_absolute_difference_sum']+=sum(abs(a-b) for a,b in zip(actual['pending_confidences'],reference['pending_confidences'],strict=True))
    compare=reports['compare']['cases'][0]['comparisons'][step+1]
    for name in dist:
        for dest,src in (('kl','kl_per_row'),('l2','relative_l2_per_row'),('top1','top1_equal_per_row')):
            dist[name][dest].extend(compare['metrics'][name][src][i] for i in pending)
    for key in envelope:
        envelope[key].extend(compare['heuristic_envelope'][key]['within'][i] for i in pending)
    rows.append(dict(step=step,pending_positions=pending,decisions=decisions))
for value in acc.values():
    value['pending_top1_agreement']=value['top1_equal']/value['pending_rows']
    value['mean_absolute_confidence_difference']=value['confidence_absolute_difference_sum']/value['pending_rows']
result=dict(state='complete',sources={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()},
    scope='One JSON W32 real native first block, 14 matched inputs. Compare pending positions only. '
          'References follow native input states, not their own controller trajectories; selected counts are counterfactual decisions, not reference throughput. '
          'Native first-block non-mask penalties are verified inert. No quality or numerical acceptance claim.',
    native_steps=native_case['steps'],rows=rows,against_native=acc,
    pending_distribution={name:dict(rows=len(value['kl']),mean_kl=statistics.mean(value['kl']),mean_l2=statistics.mean(value['l2']),top1_agreement=statistics.mean(value['top1'])) for name,value in dist.items()},
    pending_envelope={key:dict(within=sum(value),rows=len(value)) for key,value in envelope.items()},
    final_block=native_case['final_block'],visible_ids=native_case['denoise_visible_ids'])
(a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({key:result[key] for key in ('native_steps','against_native','pending_distribution','pending_envelope')},indent=2))
for name in data:
    print(name,[row['decisions'][name]['count'] for row in rows])
