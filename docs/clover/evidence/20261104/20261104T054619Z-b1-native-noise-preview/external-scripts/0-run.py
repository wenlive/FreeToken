"""Run one qualification child with immutable source and environment evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--cwd', required=True, type=Path)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = Path('C:/workspace/runs/T-20261104-clover-improve') / (stamp + '-' + a.name)
out.mkdir(parents=True, exist_ok=False)
command = a.command[1:] if a.command[:1] == ['--'] else a.command
command = [v.replace('{output}', str(out / 'data')) for v in command]

def git(*args):
    return subprocess.check_output(['git', *args], cwd=a.cwd)

patch = git('diff', '--binary', 'HEAD')
(out / 'source.patch').write_bytes(patch)
manifest = dict(started_at=stamp, command=command, cwd=str(a.cwd),
    commit=git('rev-parse', 'HEAD').decode().strip(),
    dirty=git('status', '--porcelain=v1').decode(),
    patch_sha256=hashlib.sha256(patch).hexdigest(), state='running',
    environment={k: os.environ.get(k) for k in ('PYTHONPATH', 'FREETOKEN_KERNEL_CACHE_DIR',
        'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'CLOVER_GGUF_CHECKPOINT', 'CLOVER_HF_SOURCE',
        'FREETOKEN_QWEN4EXP_MODEL', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'CUDA_VISIBLE_DEVICES',
        'FREETOKEN_DISABLE_OVERLAP_SCHEDULING')})
external = [Path(__file__), *[Path(value) for value in command if value.endswith('.py') and Path(value).is_absolute()]]
manifest['external_scripts'] = []
for index, source in enumerate(external):
    payload = source.read_bytes()
    target = out / 'external-scripts' / (str(index) + '-' + source.name)
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(payload)
    manifest['external_scripts'].append(dict(source=str(source), artifact=str(target.relative_to(out)),
        sha256=hashlib.sha256(payload).hexdigest()))
for relative in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0'):
    if relative:
        target = out / 'untracked-source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((a.cwd / relative).read_bytes())

def save():
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

print(out, flush=True)
save()
start = time.monotonic()
try:
    with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
        child = subprocess.Popen(command, cwd=a.cwd, stdout=stdout, stderr=stderr)
        manifest['pid'] = child.pid
        save()
        code = child.wait()
    manifest.update(exit_code=code, state='passed' if code == 0 else 'failed')
except BaseException as error:
    manifest.update(state='failed', error=repr(error))
    raise
finally:
    manifest['elapsed_seconds'] = time.monotonic() - start
    save()
print('exit=' + str(code), flush=True)
sys.exit(code)
