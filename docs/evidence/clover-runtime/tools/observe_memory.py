"""CPU-only host working-set observer; no CUDA calls in performance campaign."""
import argparse
import datetime
import json
from pathlib import Path
import time

import psutil

parser=argparse.ArgumentParser()
parser.add_argument('--pid',type=int,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
process=psutil.Process(args.pid)
record=dict(pid=args.pid,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Sampled process working set/private bytes plus system available RAM; no GPU polling',samples=[])
while True:
    try:
        entries=[]
        for child in [process,*process.children(recursive=True)]:
            try:
                memory=child.memory_info()
                entries.append(dict(pid=child.pid,rss=memory.rss,private=getattr(memory,'private',0)))
            except psutil.NoSuchProcess:pass
        system=psutil.virtual_memory()
        record['samples'].append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rss=sum(m['rss'] for m in entries),private=sum(m['private'] for m in entries),
            processes=entries,available=system.available))
    except psutil.NoSuchProcess:break
    time.sleep(1)
record.update(state='complete',peak_rss=max(s['rss'] for s in record['samples']),
    peak_private=max(s['private'] for s in record['samples'] if s['private'] is not None),
    minimum_system_available=min(s['available'] for s in record['samples']))
(args.output/'results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='samples'}))
