"""Group Clover tools and update current imports, preserving historical sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path.cwd().resolve()
assert root == Path('C:/workspace/wt/T-20261102-clover-q8/FreeToken').resolve()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)
a.output.mkdir(parents=True, exist_ok=False)
bench = root / 'benchmarks'
files = sorted([p for p in bench.iterdir() if p.is_file() and 'clover' in p.name])
modules = {p.stem for p in files if p.suffix == '.py'}
legacy = {'bench_clover_long.py', 'bench_clover_cpu.py'}
records = []
for source in files:
    target = root / ('docs/clover/evidence/legacy-tools' if source.name in legacy else 'benchmarks/clover') / source.name
    target = target.resolve()
    assert source.resolve().is_relative_to(root) and target.is_relative_to(root) and not target.exists()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'mv', str(source.relative_to(root)), str(target.relative_to(root))], check=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    records.append(dict(old=str(source.relative_to(root)), new=str(target.relative_to(root)), sha256=digest))
for path in list((bench / 'clover').glob('*.py')) + list((root / 'tests').rglob('*.py')):
    original = path.read_text(encoding='utf-8')
    code = original.replace('from clover.oracles import ', 'from benchmarks.clover.oracles import ')
    for module in modules:
        code = re.sub(r'(?m)^(\s*)from ' + re.escape(module) + r' import ',
            r'\1from benchmarks.clover.' + module + ' import ', code)
        code = re.sub(r'(?m)^(\s*)import ' + re.escape(module) + r'(?=\s*(?:as\s|$))',
            r'\1from benchmarks.clover import ' + module, code)
        code = code.replace('benchmarks/' + module + '.py', 'benchmarks/clover/' + module + '.py')
    if code != original:
        path.write_text(code, encoding='utf-8')
(a.output / 'results.json').write_text(json.dumps(dict(state='moved', files=records), indent=2), encoding='utf-8')
print(json.dumps(dict(moved=len(files), archived=len(legacy))))
