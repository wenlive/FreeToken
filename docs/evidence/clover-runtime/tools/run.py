"""Record each task execution independently; never discard a failed attempt."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--cwd', required=True)
    parser.add_argument('--env', action='append', default=[])
    parser.add_argument('--observe-memory', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    # A newly prioritized audit waits for the current GPU child to finish, then
    # owns the next slot. Existing experiments and their timings are untouched.
    priority_path = Path(__file__).with_name('gpu-priority.json')
    while priority_path.exists():
        try:
            priority = json.loads(priority_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            time.sleep(.1)
            continue
        if priority.get('state') != 'held' or args.name in priority.get('allowed_names', []):
            break
        time.sleep(1)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path('C:/workspace/runs/T-20261102-clover-q8') / f'{stamp}-{args.name}'
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    overrides = dict(item.split('=', 1) for item in args.env)
    overrides.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    env.update(overrides)
    identity = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=args.cwd, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain=v1'], cwd=args.cwd, capture_output=True, text=True)
    # A dirty-file list alone cannot reproduce an experiment. Preserve the exact
    # tracked patch and untracked source files before launching the child.
    patch = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=args.cwd, capture_output=True)
    (output / 'source.patch').write_bytes(patch.stdout)
    untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '-z'],
                               cwd=args.cwd, capture_output=True)
    for relative in untracked.stdout.decode('utf-8').split('\0'):
        if relative:
            target = output / 'untracked-source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(args.cwd) / relative, target)
    manifest = dict(command=command, cwd=args.cwd, environment_overrides=overrides,
                    commit=identity.stdout.strip(), dirty=status.stdout, started_at=stamp,
                    state='running', output=str(output))
    record = output / 'manifest.json'
    record.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output, flush=True)
    start = time.monotonic()
    try:
        with (output / 'stdout.log').open('wb') as stdout, (output / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, cwd=args.cwd, env=env, stdout=stdout, stderr=stderr)
            manifest['pid'] = process.pid
            record.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            observer = None
            if args.observe_memory:
                observer = subprocess.Popen([
                    sys.executable, str(Path(__file__).with_name('observe_memory.py')),
                    '--pid', str(process.pid), '--output', str(output / 'host-memory')],
                    stdout=stdout, stderr=stderr)
            code = process.wait()
            if observer is not None and observer.wait() != 0:
                raise RuntimeError('Host memory observer failed')
    except BaseException as exc:
        manifest.update(state='failed', error=repr(exc))
        raise
    else:
        manifest.update(state='passed' if code == 0 else 'failed', exit_code=code)
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - start
        record.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'exit={code} elapsed={manifest["elapsed_seconds"]:.2f}s', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())

