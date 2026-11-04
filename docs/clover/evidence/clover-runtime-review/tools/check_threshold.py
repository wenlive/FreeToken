"""Compare real threshold outputs and acceptance traces, excluding wall timers."""
import hashlib
import json
from pathlib import Path

root=Path('C:/workspace/runs/T-20261102-clover-review')
result=dict(state='complete',scope='Count/JSON, selfspec8/16, cap64, P1; correctness supplement, not ABAB timings',cases={})
for label in ('q8','nvfp4'):
 paths=[root/f'{label}-threshold-correctness-{side}'/'results.json' for side in ('before','after')]
 before,after=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
 assert before['state']==after['state']=='complete'
 checks=[]
 for a,b in zip(before['runs'],after['runs'],strict=True):
  keys=('prompt','mode','width','repeat','warmup')
  assert all(a[k]==b[k] for k in keys)
  extract=lambda row:[(r['accepted_drafts'],r['emitted_tokens']) for r in row['rounds']]
  check=dict(**{k:a[k] for k in keys},tokens_equal=a['token_ids']==b['token_ids'],
             acceptance_trace_equal=extract(a)==extract(b))
  checks.append(check)
 assert all(r['tokens_equal'] and r['acceptance_trace_equal'] for r in checks)
 result['cases'][label]=dict(checks=checks,inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
(root/'final-threshold.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({label:dict(total_pairs=len(data['checks']),measured_pairs=sum(not r['warmup'] for r in data['checks'])) for label,data in result['cases'].items()}))
