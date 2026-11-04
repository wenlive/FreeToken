"""Verify this evidence batch after checkout, without importing the model runtime."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = json.loads((root/'inventory.json').read_text(encoding='utf-8-sig'))['files']
actual = {path.relative_to(root).as_posix(): path for path in root.rglob('*')
          if path.is_file() and path != root/'inventory.json'}
errors = sorted(set(actual) ^ set(expected))
for name in actual.keys() & expected.keys():
    path, receipt = actual[name], expected[name]
    if (path.stat().st_size != receipt['bytes']
            or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']):
        errors.append(name)
print(json.dumps(dict(files=len(actual), mismatches=errors)))
raise SystemExit(bool(errors))
