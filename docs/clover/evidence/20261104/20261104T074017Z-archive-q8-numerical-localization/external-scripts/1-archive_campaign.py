"""Append completed run evidence; retain large binary operands in the local run store."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--destination', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path('C:/workspace/runs/T-20261104-clover-improve')
a.destination.mkdir(parents=True, exist_ok=True)
text_types = {'.json','.log','.py','.patch','.xml','.md','.txt','.cpp','.ps1','.toml','.yaml','.yml'}
catalog = dict(version=1, source_root=str(root), runs=[])
entries = []


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 << 20),b''):
            result.update(block)
    return result.hexdigest()


for directory in sorted(p for p in root.iterdir() if p.is_dir()):
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.exists() else None
    if manifest is not None and manifest['state'] == 'running':
        continue
    catalog['runs'].append(dict(run=directory.name,
        process_state=manifest['state'] if manifest else 'legacy_without_wrapper_manifest',
        commit=manifest['commit'] if manifest else None,
        elapsed_seconds=manifest.get('elapsed_seconds') if manifest else None))
    for source in sorted(p for p in directory.rglob('*') if p.is_file()):
        relative = source.relative_to(root).as_posix()
        size, sha = source.stat().st_size, digest(source)
        item = dict(source=relative, size=size, sha256=sha)
        if source.suffix not in text_types:
            item['storage'] = 'external'
            entries.append(item)
            continue
        compressed = size > 4 << 20
        target = a.destination / (relative + ('.gz' if compressed else ''))
        content = source.read_bytes()
        payload = gzip.compress(content,compresslevel=1,mtime=0) if compressed else content
        if target.exists():
            assert target.read_bytes() == payload, f'Existing evidence changed: {target}'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)
        item.update(storage='gzip' if compressed else 'raw',
            archive=target.relative_to(a.destination).as_posix(), archive_size=len(payload),
            archive_sha256=hashlib.sha256(payload).hexdigest())
        entries.append(item)
catalog['files'] = entries
catalog['counts'] = dict(runs=len(catalog['runs']), files=len(entries),
    archived_bytes=sum(item.get('archive_size',0) for item in entries),
    external_bytes=sum(item['size'] for item in entries if item['storage']=='external'))
(a.destination/'inventory.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
(a.output/'results.json').write_text(json.dumps(dict(state='complete',
    destination=str(a.destination), counts=catalog['counts']),indent=2),encoding='utf-8')
print(json.dumps(catalog['counts'],indent=2))
