"""Inventory the task overlay without initializing CUDA or changing dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
prefix=Path(sys.prefix)
site=prefix/'Lib/site-packages'
def fingerprint(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
result=dict(state='complete',observed_at=datetime.now(timezone.utc).isoformat(),executable=sys.executable,
    python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,
    source_mode='Explicit per-process PYTHONPATH; immutable dependency site from previous verified environment; no editable install',
    packages=sorted([dict(name=d.metadata['Name'],version=d.version,path=str(d.locate_file('')))
        for d in importlib.metadata.distributions()],key=lambda d:(d['name'].lower(),d['path'])),
    modules={name:importlib.util.find_spec(name).origin for name in ('torch','triton','transformers','numpy','gguf','pytest')},
    pth=[dict(**fingerprint(path),content=path.read_text(encoding='utf-8')) for path in site.glob('*.pth')],
    native_extensions=[])
for root in (Path('C:/workspace/wt/T-20261102-clover-q8/FreeToken/python/freetoken'),
    Path('C:/workspace/builds/T-20261102-clover-q8/torch-extensions')):
    result['native_extensions'].extend(fingerprint(path) for path in sorted(root.rglob('*.pyd')))
assert 'torch.cuda' not in sys.modules
(args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(state='complete',packages=len(result['packages']),native_extensions=len(result['native_extensions']))))
