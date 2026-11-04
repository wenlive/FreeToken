"""Verify archived bytes; optionally also check locally retained binary operands."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--verify-external', action='store_true')
p.add_argument('--tracked', action='store_true', help='Require identical blobs in the Git index')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
inventory = json.loads((root/'inventory.json').read_text())
if a.tracked:
    repo = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip())
    object_format = subprocess.check_output(['git','rev-parse','--show-object-format'],cwd=root,text=True).strip()
    index = {}
    for record in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=repo).split(b'\0'):
        if record:
            header,name = record.split(b'\t',1)
            index[name.decode()] = header.split()[1].decode()
checked = external = 0
for item in inventory['files']:
    if item['storage'] == 'external':
        if not a.verify_external:
            external += 1
            continue
        path = Path(inventory['source_root'])/item['source']
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 << 20),b''):
                size += len(block)
                digest.update(block)
        assert size == item['size'] and digest.hexdigest() == item['sha256'], path
    else:
        path = root/item['archive']
        payload = path.read_bytes()
        if a.tracked:
            object_bytes = b'blob ' + str(len(payload)).encode() + b'\0' + payload
            assert index.get(path.relative_to(repo).as_posix()) == hashlib.new(object_format,object_bytes).hexdigest(), path
        assert len(payload) == item['archive_size']
        assert hashlib.sha256(payload).hexdigest() == item['archive_sha256'], path
        content = gzip.decompress(payload) if item['storage']=='gzip' else payload
        assert len(content) == item['size'] and hashlib.sha256(content).hexdigest() == item['sha256'], path
    checked += 1
print(f'Verified {checked} files; {external} external binary files not requested')
