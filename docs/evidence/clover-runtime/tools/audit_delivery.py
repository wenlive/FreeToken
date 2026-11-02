"""Verify recorded bytes, output bindings and staged evidence portability."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--evidence',type=Path,required=True)
p.add_argument('--checkout',type=Path)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
def read(name):
    return json.loads((args.evidence/(name+'.json')).read_text(encoding='utf-8'))
inventory=read('inventory')['files']
for name,record in inventory.items():
    raw=(args.evidence/name).read_bytes()
    assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
sources=read('sources')
raw_checked=0
for name,record in sources.items():
    if 'sha256' not in record:
        continue
    raw=Path(record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['sha256'],name
    raw_checked+=1
outputs=read('outputs')
for digest,value in outputs.items():
    raw=json.dumps([value['token_ids'],value['text']],ensure_ascii=False,separators=(',',':')).encode()
    assert hashlib.sha256(raw).hexdigest()==digest
matrices=read('matrices')
requests=0
for label,matrix in matrices.items():
    groups={}
    for row in matrix['runs']:
        assert row['output_sha256'] in outputs,(label,row)
        group=(row['prompt'],row['mode'],row['width'])
        groups.setdefault(group,[]).append(row)
    for group,rows in groups.items():
        assert len(rows)==7 and sum(r['warmup'] for r in rows)==2,(label,group)
        assert sum(not r['warmup'] for r in rows)==5,(label,group)
    requests+=len(matrix['runs'])
acceptance=read('acceptance')
assert all(acceptance[k] for k in ('T1','T2','T3_conservative','T4','C_preview_state_unchanged',
    'C_graph_vs_eager_qualified_inputs','C_causal_not_below_previous','C_selfspec_aggregate','R_with_documented_skips','N_speed_all_25'))
assert not acceptance['C_denoise_aggregate'] and len(acceptance['C_denoise_failed'])==4
assert len(acceptance['N_W32_numeric_reproduction_remaining'])==2
staged=0
if args.checkout:
    relative=args.evidence.resolve().relative_to(args.checkout.resolve()).as_posix()
    listing=subprocess.check_output(['git','-C',str(args.checkout),'ls-files','-s','-z','--',relative])
    index={}
    for entry in listing.split(b'\0'):
        if entry:
            meta,path=entry.split(b'\t',1)
            index[path.decode()]=meta.split()[1].decode()
    for path in args.evidence.rglob('*'):
        if path.is_file():
            key=path.resolve().relative_to(args.checkout.resolve()).as_posix()
            raw=path.read_bytes()
            digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            assert index.get(key)==digest,('Git changed evidence bytes',key)
            staged+=1
result=dict(state='complete',bundle_files=len(inventory),original_raw_sources=raw_checked,
    unique_outputs=len(outputs),formal_matrix_requests=requests,staged_verbatim_files=staged,
    acceptance=acceptance,note='Recorded acceptance failures are preserved, not changed to pass')
(args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='acceptance'}))
