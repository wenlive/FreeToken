"""Read-only source/evidence audit; does not import Torch or run the model."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
repo = Path('C:/workspace/wt/T-20261102-clover-q8/FreeToken')
nvfp4 = Path('C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken')
before = repo.with_name('FreeToken-q8-runtime-before')
evidence = repo / 'docs/evidence/clover-runtime'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def git(path, *command):
    return subprocess.check_output(['git', '-C', str(path), *command], text=True,
                                   encoding='utf-8').strip()


def span(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


inventory = read(evidence / 'inventory.json')
errors = []
for relative, receipt in inventory['files'].items():
    data = (evidence / relative).read_bytes()
    if len(data) != receipt['bytes'] or hashlib.sha256(data).hexdigest() != receipt['sha256']:
        errors.append(relative)

matrices = read(evidence / 'matrices.json')
benchmarks = {}
for name in ('q8_runtime_before', 'q8_p1', 'nvfp4_before', 'nvfp4_after'):
    source = matrices[name]
    cases = {}
    for prompt in ('Count', 'Sky', 'Python', 'JSON', 'Chinese'):
        rows = [r for r in source['runs'] if r['prompt'] == prompt
                and r['mode'] == 'causal' and not r['warmup']]
        cases[prompt] = {
            key: span([r[key] for r in rows]) for key in
            ('ttft_seconds', 'decode_tokens_per_second', 'end_to_end_tokens_per_second')
        }
        cases[prompt]['logical_h2d_bytes'] = sorted({r['decode_expert_copy_payload_bytes'] for r in rows})
        cases[prompt]['output_digests'] = sorted({r['output_sha256'] for r in rows})
        cases[prompt]['reserved_bytes'] = sorted({r['reserved'] for r in rows if 'reserved' in r})
    benchmarks[name] = dict(startup_seconds=source['startup_seconds'],
                            allocated_peak=source.get('cuda_peak_allocated'),
                            reserved_peak=source.get('cuda_peak_reserved'), cases=cases)


def methods(path, selected):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in selected}


relative_runtime = 'python/freetoken/models/clover_moe/runtime.py'
selected = ('process_logits', 'state', '_penalize_verify')
old_methods = methods(before / relative_runtime, selected)
new_methods = methods(repo / relative_runtime, selected)
header = read(Path('C:/workspace/tasks/T-20261102-clover-q8/official-q8-header.json'))
meta = header['metadata']
linear_layers = sum(meta['clover_moe.attention.recurrent_layers'])
nk = meta['clover_moe.ssm.group_count']
nv = meta['clover_moe.ssm.time_step_rank']
dk = meta['clover_moe.ssm.state_size']
inner = meta['clover_moe.ssm.inner_size']
conv = meta['clover_moe.ssm.conv_kernel']
state_row = linear_layers * (nv * dk * (inner // nv) * 4 + (2 * nk * dk + inner) * (conv - 1) * 2)
hf = read(evidence / 'hf-initial-preview.json')
preview_fields = ('prompt', 'width', 'step', 'mean_kl', 'argmax_agreement', 'threshold_crossing_rows')
report = dict(
    state='complete', observed_at_utc=datetime.now(timezone.utc).isoformat(),
    executable=sys.executable,
    environment={name: importlib.metadata.version(name) for name in
                 ('torch', 'triton-windows', 'gguf', 'transformers', 'pytest')},
    sources={name: dict(path=str(path), commit=git(path, 'rev-parse', 'HEAD'),
                       status=git(path, 'status', '--porcelain=v1'))
             for name, path in [('q8', repo), ('nvfp4', nvfp4), ('q8_before_control', before)]},
    inventory=dict(checked_files=len(inventory['files']), mismatches=errors),
    unchanged_causal_methods={name: old_methods.get(name) == new_methods.get(name) for name in selected},
    shared_runtime_range_diff=[
        git(repo, 'range-diff', '69acf2f^..69acf2f', '40e399f^..40e399f'),
        git(repo, 'range-diff', '25fb44b^..25fb44b', '40f96c0^..40f96c0')],
    historical_causal_measurements=benchmarks,
    fixed_input_preview=[{key: row[key] for key in preview_fields} for row in hf['cases']],
    derived_geometry=dict(linear_layers=linear_layers, full_attention_layers=len(meta['clover_moe.attention.recurrent_layers'])-linear_layers,
                          transaction_row_bytes_bf16_conv_fp32_ssm=state_row,
                          transaction_bytes={str(w): w * state_row + 5 for w in (8, 16, 32)},
                          note='Geometry calculation, not a new allocation measurement'),
    limitations='No model inference, GPU timing, full model rehash or performance requalification performed.',
)
(args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({key: report[key] for key in ('state', 'inventory', 'unchanged_causal_methods', 'derived_geometry')}, indent=2))
if errors:
    raise SystemExit(1)
