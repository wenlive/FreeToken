"""Compare all four startup turns and verify complete loaded tensor fingerprints."""
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
result=dict(state='running',turns=[])
data=[]
for turn in range(1,5):
    paths=[p for p in root.glob(f'*-c5-startup-abab-{turn}/manifest.json')
        if json.loads(p.read_bytes())['state']=='passed']
    assert len(paths)==1,paths
    manifest=json.loads(paths[0].read_bytes())
    path=paths[0].parent/'data/results.json'
    value=json.loads(path.read_bytes())
    assert value['state']=='complete'
    reader=[call['seconds'] for call in value['calls'] if call['operation']=='GGUFReader']
    data.append(value)
    result['turns'].append(dict(turn=turn,variant='before' if turn%2 else 'after',
        commit=manifest['commit'],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        startup_seconds=value['startup_seconds'],reader_count=len(reader),reader_seconds=reader,
        reader_total_seconds=sum(reader),settings=value.get('settings'),
        manifest=manifest))
before,after=data[0]['loaded_tensor_hashes'],data[1]['loaded_tensor_hashes']
result['tensor_names_equal']=before.keys()==after.keys()
result['tensor_hash_mismatches']=[name for name in before.keys()|after.keys() if before.get(name)!=after.get(name)]
result['tensor_count']=len(before)
captured_settings=[value['settings'] for value in data if 'settings' in value]
result['settings_captured_turns']=[i+1 for i,value in enumerate(data) if 'settings' in value]
result['settings_equal']=(all(value==captured_settings[0] for value in captured_settings)
    if len(captured_settings)==4 else None)
result['settings_note']='Early profile revision did not save resolved settings; preserve the original result and command/script identity rather than inventing metadata.'
result['startup_pairs']=[dict(before=data[i]['startup_seconds'],after=data[i+1]['startup_seconds'],
    reduction_percent=100*(1-data[i+1]['startup_seconds']/data[i]['startup_seconds'])) for i in (0,2)]
result['startup_medians']=dict(before=statistics.median(data[i]['startup_seconds'] for i in (0,2)),
    after=statistics.median(data[i]['startup_seconds'] for i in (1,3)))
result['scope']='Same-boot ABAB; full per-turn times retained, including first-use compilation. '
result['scope']+='Weight hashing occurs after the startup timer. Only reader durations are summed; nested parser timers are not.'
result['state']='complete' if result['tensor_names_equal'] and not result['tensor_hash_mismatches'] else 'failed'
(a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
assert result['state']=='complete'
