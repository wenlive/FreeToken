"""Run one recorded experiment with source provenance and retained failures."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--cwd', type=Path, required=True)
p.add_argument('--env', action='append', default=[])
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
handoff = Path('C:/workspace/tasks/T-20261102-clover-review/campaign-handoff.json')
if handoff.exists() and a.name in json.loads(handoff.read_text(encoding='utf-8'))['stop_names']:
    print('Controlled campaign handoff before starting this GPU job', flush=True)
    raise SystemExit(2)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = Path('C:/workspace/runs/T-20261102-clover-review') / f'{stamp}-{a.name}'
out.mkdir(parents=True, exist_ok=False)
command = a.command[1:] if a.command[:1] == ['--'] else a.command
def git(*args):
    return subprocess.check_output(['git', '-C', str(a.cwd), *args])
patch = git('diff', '--binary', 'HEAD')
(out / 'source.patch').write_bytes(patch)
env = dict(os.environ)
overrides = dict(item.split('=', 1) for item in a.env)
env.update(overrides)
untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode('utf-8').split('\0')
for relative in filter(None, untracked):
    target = out / 'untracked-source' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((a.cwd / relative).read_bytes())
manifest = dict(name=a.name, started_at_utc=stamp, state='running', command=command,
                cwd=str(a.cwd), commit=git('rev-parse', 'HEAD').decode().strip(),
                dirty=git('status', '--porcelain=v1').decode('utf-8'),
                native_extensions={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (a.cwd/'python/freetoken/kernel').glob('*.pyd')},
                patch_sha256=hashlib.sha256(patch).hexdigest(), environment_overrides=overrides)
manifest['script_receipts'] = {}
manifest['production_inventory'] = {
    str(path.relative_to(a.cwd)).replace('\\','/'): hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    for path in (a.cwd/'python/freetoken').rglob('*')
    if path.is_file() and path.suffix in ('.py','.cu','.cuh','.cpp','.h')
}
for argument in command:
    path = Path(argument)
    if path.suffix.lower() in ('.py', '.ps1'):
        path = path if path.is_absolute() else a.cwd/path
        if path.is_file():
            manifest['script_receipts'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
def save():
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
def device_snapshot():
    observed = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.used,pstate,temperature.gpu,power.draw,clocks.current.sm',
        '--format=csv'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    return dict(exit_code=observed.returncode,output=observed.stdout.strip())
manifest['device_before'] = device_snapshot()
save()
start = time.monotonic()
try:
    with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
        child = subprocess.Popen(command, cwd=a.cwd, env=env, stdout=stdout, stderr=stderr)
        manifest['pid'] = child.pid
        save()
        code = child.wait()
    manifest.update(state='passed' if code == 0 else 'failed', exit_code=code)
finally:
    manifest['elapsed_seconds'] = time.monotonic() - start
    manifest['device_after'] = device_snapshot()
    save()
print(json.dumps(dict(output=str(out), **{k:manifest[k] for k in ('state','elapsed_seconds')})), flush=True)
sys.exit(code)
