"""Check shipped arithmetic, immutable baselines, artifact digests and regressions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rehash-model',action='store_true')
a=p.parse_args()
root=Path('C:/workspace/runs/T-20261102-clover-review')
q8=Path('C:/workspace/wt/T-20261102-clover-q8/FreeToken')
nv=Path('C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken')
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True,encoding='utf-8').strip()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(16<<20),b''):h.update(block)
 return h.hexdigest()
report=dict(state='complete',heads={label:git(path,'rev-parse','HEAD') for label,path in [('q8',q8),('nvfp4',nv)]})
model=Path('C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf')
report['official_q8']=dict(path=str(model),bytes=model.stat().st_size)
assert model.stat().st_size==37863343264
if a.rehash_model:
 report['official_q8']['sha256']=digest(model)
 assert report['official_q8']['sha256']=='834529e7aae000305ae94656782726bcf0ed77976f95a56383447d8903afddab'
frozen=Path('C:/workspace/wt/T-20261102-clover-review')
report['frozen_source_status']={p.name:git(p,'status','--porcelain=v1') for p in frozen.glob('FreeToken-*')}
assert not any(report['frozen_source_status'].values())
shared=['engine/speculative.py','kernel/triton/clover_verify.py','kernel/triton/clover_store.py',
        'kernel/triton/clover_attention.py','models/clover_moe/graph.py',
        'models/clover_moe/model.py','models/clover_moe/preview.py']
report['shared_arithmetic_equal']={r:(q8/'python/freetoken'/r).read_bytes().replace(b'\r\n',b'\n') ==
 (nv/'python/freetoken'/r).read_bytes().replace(b'\r\n',b'\n') for r in shared}
assert all(report['shared_arithmetic_equal'].values())
def common_runtime(path):
 tree=ast.parse((path/'python/freetoken/models/clover_moe/runtime.py').read_text(encoding='utf-8'))
 cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='CloverRuntime')
 return {node.name:ast.dump(node,include_attributes=False) for node in cls.body
  if isinstance(node,ast.FunctionDef) and node.name!='__init__'}
report['shared_runtime_methods_equal']=common_runtime(q8)==common_runtime(nv)
report['runtime_constructor_exception']='Q8 initializes generation from GGUF metadata; NVFP4 reads its HF config/generation JSON.'
assert report['shared_runtime_methods_equal']
evidence=q8/'docs/evidence/clover-runtime'
old=read(evidence/'inventory.json')
report['r0_inventory']={r:(evidence/r).stat().st_size==m['bytes'] and digest(evidence/r)==m['sha256']
 for r,m in old['files'].items()}
assert all(report['r0_inventory'].values())
report['real_regressions']={}
report['delivered_source_matches_final_real_smoke']={}
for label,source in [('q8',q8),('nvfp4',nv)]:
 manifests=sorted(root.glob(f'202*-qwen35-{label}-review/manifest.json'))
 assert manifests
 receipt=read(manifests[-1])
 current={str(path.relative_to(source)).replace('\\','/'):hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
  for path in (source/'python/freetoken').rglob('*')
  if path.is_file() and path.suffix in ('.py','.cu','.cuh','.cpp','.h')}
 equal=current==receipt['production_inventory']
 native={path.name:digest(path) for path in (source/'python/freetoken/kernel').glob('*.pyd')}
 assert equal and native==receipt['native_extensions']
 report['delivered_source_matches_final_real_smoke'][label]=dict(equal=True,manifest=str(manifests[-1]),
  production_files=len(current),native_extensions=native)
def compare(current,previous):
 assert current['state']==previous['state']=='complete'
 assert current['protected_before']==current['protected_after']==previous['protected_after']
 oracle={r['prompt']:r for r in previous['cases']}
 assert set(oracle)=={r['prompt'] for r in current['cases']}
 checks={}
 for case in current['cases']:
  old=oracle[case['prompt']]
  checks[case['prompt']]={field:case[field]==old[field]
   for field in ('prompt_ids','token_ids','rounds') if field in old}
 assert all(all(row.values()) for row in checks.values()),checks
 return dict(state='passed',cases=checks,protected_metadata_unchanged=True)
for label in ('q8','nvfp4'):
 for mode in ('off','on'):
  name=f'mtp-{label}-review-{mode}'
  current=read(root/name/'results.json')
  previous_path=Path(f'C:/workspace/runs/T-20261102-clover-q8/mtp-main-{mode}/results.json')
  previous=read(previous_path)
  report['real_regressions'][name]=compare(current,previous)
  report['real_regressions'][name]['input_receipts']=dict(current=f'runs/{name}/data/results.json',
   current_sha256=digest(root/name/'results.json'),previous_main=f'audit/main-controls/mtp-main-{mode}.json',
   previous_sha256=digest(previous_path))
 name=f'qwen35-{label}-review'
 current=read(root/name/'results.json')
 previous_path=Path('C:/workspace/runs/T-20261102-clover-q8/qwen35-real-nvfp4-main/results.json')
 previous=read(previous_path)
 report['real_regressions'][name]=compare(current,previous)
 report['real_regressions'][name]['input_receipts']=dict(current=f'runs/{name}/data/results.json',
  current_sha256=digest(root/name/'results.json'),previous_main='audit/main-controls/qwen35-real-nvfp4-main.json',
  previous_sha256=digest(previous_path))
(root/'final-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(state=report['state'],shared_equal=all(report['shared_arithmetic_equal'].values()),
 r0_files=len(report['r0_inventory']),official_q8=report['official_q8'])))
