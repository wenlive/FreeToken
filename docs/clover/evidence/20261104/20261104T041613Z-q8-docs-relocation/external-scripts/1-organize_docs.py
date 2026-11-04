"""Move Clover records without rewriting any historical evidence bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
root = Path.cwd().resolve()
assert root in [Path(v).resolve() for v in (
    'C:/workspace/wt/T-20261102-clover-q8/FreeToken',
    'C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken')]
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root), 'Expected clean source tree'
plans = [('docs/' + name, 'docs/clover/' + name) for name in (
    'clover-performance.md', 'clover-runtime-performance.md', 'clover-runtime-review.md')]
plans += [('docs/evidence/' + name, 'docs/clover/evidence/' + name)
    for name in ('clover-runtime', 'clover-runtime-review')]
a.output.mkdir(parents=True, exist_ok=False)
records = []
for old, new in plans:
    source, target = (root / old).resolve(), (root / new).resolve()
    assert source.is_relative_to(root) and target.is_relative_to(root)
    assert source.exists() and not target.exists(), (source, target)
    payloads = list(source.rglob('*')) if source.is_dir() else [source]
    hashes = {str(f.relative_to(source)) if f != source else '': hashlib.sha256(f.read_bytes()).hexdigest()
        for f in payloads if f.is_file()}
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'mv', old, new], cwd=root, check=True)
    for name, checksum in hashes.items():
        f = target / name if name else target
        assert hashlib.sha256(f.read_bytes()).hexdigest() == checksum, f
    records.append(dict(old=old, new=new, files=hashes))
readme = root / 'benchmarks/README.md'
raw = readme.read_bytes()
readme.write_bytes(raw.replace(b'../docs/clover-performance.md', b'../docs/clover/clover-performance.md'))
for name in ('clover-runtime', 'clover-runtime-review'):
    archive = root / 'docs/clover/evidence' / name
    inventory = json.loads((archive / 'inventory.json').read_text(encoding='utf-8-sig'))['files']
    for relative, receipt in inventory.items():
        payload = (archive / relative).read_bytes()
        assert len(payload) == receipt['bytes']
        assert hashlib.sha256(payload).hexdigest() == receipt['sha256'], relative
(a.output / 'results.json').write_text(json.dumps(dict(state='passed', moves=records,
    historical_bytes_unchanged=True, inventories_verified=True), indent=2), encoding='utf-8')
print(json.dumps(dict(state='passed', moved_files=sum(len(r['files']) for r in records))))
