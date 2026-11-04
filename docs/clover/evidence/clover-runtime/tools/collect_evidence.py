"""Build a compact reviewable evidence bundle only from completed recorded runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import statistics

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = Path('C:/workspace/runs/T-20261102-clover-q8')
sources, outputs = {}, {}
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def read(name, filename='results.json'):
    path = root / name / filename
    raw = path.read_bytes()
    value = json.loads(raw)
    assert value.get('state', 'complete') in ('complete', 'passed'), (name, value.get('state'))
    receipt = dict(path=str(path), sha256=sha(raw), bytes=len(raw))
    # A suffix also matches stock-q8-* for q8-*; bind the exact run label.
    manifests = sorted(d for d in root.glob('20*-' + name) if d.name.split('-',1)[1] == name)
    for directory in manifests[-1:]:
        mpath = directory / 'manifest.json'
        if mpath.exists():
            manifest = json.loads(mpath.read_text(encoding='utf-8'))
            assert manifest['state'] == 'passed', (name, manifest)
            receipt['run'] = manifest
            memory = directory / 'host-memory/results.json'
            if memory.exists():
                observed = json.loads(memory.read_text(encoding='utf-8'))
                receipt['host_memory'] = {k: v for k, v in observed.items() if k != 'samples'}
    sources[name + '/' + filename] = receipt
    return value
def save(name, value):
    (args.output / (name + '.json')).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def optional(name):
    directories = sorted(d for d in root.glob('20*-' + name) if d.name.split('-',1)[1] == name)
    assert directories, ('optional probe was not attempted', name)
    manifest = json.loads((directories[-1] / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['state'] in ('passed', 'failed'), (name, manifest['state'])
    if manifest['state'] == 'passed':
        return dict(state='complete', data=compact(read(name)))
    sources[name] = dict(run=manifest)
    return dict(state='failed', run=manifest, excluded_from_performance=True)

def output_comparison(before, after):
    previous = {(r['prompt'], r['mode'], r['width'], r['repeat']): r
        for r in before['runs'] if not r['warmup']}
    rows = []
    for r in after['runs']:
        if r['warmup']:
            continue
        old = previous.get((r['prompt'], r['mode'], r['width'], r['repeat']))
        # Diagnostic one-repeat runs number their measured request 1 rather than 2.
        if old is None:
            old = next(x for x in before['runs'] if not x['warmup'] and key(x) == key(r))
        rows.append(dict(prompt=r['prompt'], mode=r['mode'], width=r['width'], repeat=r['repeat'],
            same_token_ids=old['token_ids'] == r['token_ids'], old_tokens=len(old['token_ids']), new_tokens=len(r['token_ids']),
            first_difference=next((i for i,(a,b) in enumerate(zip(old['token_ids'],r['token_ids'])) if a!=b),
                min(len(old['token_ids']),len(r['token_ids'])) if len(old['token_ids'])!=len(r['token_ids']) else None)))
    return rows
def key(row):
    return row['prompt'], row['mode'], row['width'] if row['mode'] != 'causal' else 1
def lookup(matrix):
    return {key(row): row for row in matrix['summary']}
def compact(matrix):
    result = {k: copy.deepcopy(v) for k, v in matrix.items() if k != 'runs'}
    result['runs'] = []
    for row in matrix['runs']:
        out = {k: v for k, v in row.items() if k not in ('text', 'token_ids', 'token_arrival_seconds', 'stream_event_arrival_seconds')}
        token_ids = row.get('token_ids')
        text = row.get('text', '')
        identity = sha(json.dumps([token_ids, text], ensure_ascii=False, separators=(',', ':')).encode())
        outputs[identity] = dict(token_ids=token_ids, text=text)
        out['output_sha256'] = identity
        result['runs'].append(out)
    return result

names = dict(q8_p0='official-q8-p0-full-matrix', q8_p1='official-q8-p1-full-matrix',
    q8_threshold='official-q8-p0-threshold-matrix', q8_threshold_cold='q8-cold-cache-p0-threshold-denoise',
    stock_mmap_strict='official-q8-p0-server-baseline',
    stock_mmap_entropy='stock-q8-p0-entropy-matrix',
    stock_none_causal='stock-q8-none-causal-audit', stock_none_selfspec='stock-q8-none-p0-selfspec',
    stock_entropy='stock-q8-none-p0-entropy', nvfp4_before='nvfp4-before-p1-full-matrix',
    nvfp4_after='nvfp4-final-p1-full-matrix', q8_runtime_before='q8-runtime-before-p1-full-matrix')
matrices = {label: read(name) for label, name in names.items()}
causal_none, spec_none = matrices.pop('stock_none_causal'), matrices.pop('stock_none_selfspec')
assert all(m['settings']['load_mode'] == 'none' and m['settings']['cache_mib'] == 8160 for m in (causal_none, spec_none))
primary = copy.deepcopy(causal_none)
primary.update(runs=causal_none['runs'] + spec_none['runs'], summary=causal_none['summary'] + spec_none['summary'],
    backend_receipts=causal_none.get('backend_receipts', []) + spec_none.get('backend_receipts', []),
    derived_from=['stock-q8-none-causal-audit/results.json', 'stock-q8-none-p0-selfspec/results.json'])
primary['settings'].update(modes='causal,selfspec:8,selfspec:16', output='Derived combined dataset; original commands and output paths retained in sources')
matrices['stock_strict'] = primary
for label, matrix in matrices.items():
    assert matrix['settings']['warmups'] == 2 and matrix['settings']['repeats'] == 5, label
matrices['q8_threshold_cold']['cache_scope'] = 'Explicitly reset GPU expert cache before every request; reset and synchronization included conservatively in timing'
native, strict, entropy, threshold = [lookup(matrices[label]) for label in ('q8_p0', 'stock_strict', 'stock_entropy', 'q8_threshold_cold')]
comparison = dict(T1=[], T2=[], T3=[], T4=[])
ordinary = lookup(read('stock-q8-none-causal-audit'))
metric = 'end_to_end_tokens_per_second'
for prompt in ('Count', 'Sky', 'Python', 'JSON', 'Chinese'):
    ft, stock = native[prompt, 'causal', 1][metric], strict[prompt, 'causal', 1][metric]
    comparison['T1'].append(dict(prompt=prompt, freetoken=ft, stock=stock, ratio=ft / stock, passed=ft >= stock))
    plain = ordinary[prompt, 'causal', 1][metric]
    comparison['T1'][-1].update(stock_ordinary_load=plain, passed_ordinary_load=ft >= plain,
        stock_mmap=lookup(matrices['stock_mmap_strict'])[prompt, 'causal', 1][metric],
        ratio_vs_best_loading=ft / max(stock, lookup(matrices['stock_mmap_strict'])[prompt, 'causal', 1][metric]))
    for width in (8, 16):
        for rule, a, b, mode in [('exact', native, strict, 'selfspec'), ('threshold', threshold, entropy, 'official')]:
            ft, stock = a[prompt, 'selfspec', width][metric], b[prompt, mode, width][metric]
            comparison['T2'].append(dict(prompt=prompt, width=width, rule=rule,
                freetoken=ft, stock=stock, ratio=ft / stock, passed=ft >= stock,
                cache_scope='resident / resident' if rule=='exact' else 'native explicit cold reset / fresh stock CLI'))
            if rule == 'threshold':
                comparison['T2'][-1]['native_resident'] = lookup(matrices['q8_threshold'])[prompt,'selfspec',width][metric]
    ft = native[prompt, 'denoise', 16][metric]
    cold_ft = threshold[prompt, 'denoise', 16][metric]
    same_width = max(strict[prompt, 'selfspec', 16][metric], entropy[prompt, 'official', 16][metric])
    best_any = max(source[prompt, mode, width][metric] for source, mode in [(strict, 'selfspec'), (entropy, 'official')] for width in (8, 16))
    comparison['T3'].append(dict(prompt=prompt, freetoken=ft, stock_best_W16=same_width,
        freetoken_cold=cold_ft,
        stock_best_any_width=best_any, ratio_vs_W16=ft / same_width, ratio_vs_any_width=ft / best_any,
        conservative_ratio=min(ft,cold_ft)/best_any,
        passed=ft >= strict[prompt,'selfspec',16][metric] and cold_ft >= entropy[prompt,'official',16][metric],
        passed_stricter_any_width=min(ft,cold_ft) >= best_any))
preview = read('q8-five-prompt-preview')
stock_preview = read('stock-q8-none-preview-perf')
sp = {(r['prompt'], r['width']): r for r in stock_preview['cases']}
for row in preview['cases']:
    other = sp[row['prompt'], row['width']]
    causal = preview['causal_median_ms'][row['prompt']]
    probe_stdout = root / 'stock-q8-none-preview-perf' / f"{row['prompt']}-W{row['width']}" / 'stdout.log'
    probe_inputs = []
    for line in probe_stdout.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if parsed.get('type') == 'input':
            probe_inputs.append(parsed)
    assert len(probe_inputs) == 1 and probe_inputs[0]['anchor'] == row['anchor'], (row['prompt'], 'preview anchor mismatch')
    comparison['T4'].append(dict(prompt=row['prompt'], width=row['width'], native_ms=row['median_ms'],
        stock_decode_ms=other['median_decode_ms'], stock_wall_ms=other['median_wall_ms'], causal_ms=causal,
        anchor=row['anchor'], same_anchor=True,
        passed_stock=row['median_ms'] <= other['median_decode_ms'], passed_2x_causal=row['median_ms'] <= 2 * causal))

correctness = dict(q8_graph=read('official-q8-graph-audit-mmq-fixed'), nvfp4_graph=read('nvfp4-final-graph-state-audit'),
    nvfp4_preview=read('nvfp4-five-prompt-preview'),
    hf_causal=read('q8-hf-forced-history-priority'), hf_selfspec=read('q8-hf-selfspec-progress'),
    hf_denoise=read('q8-hf-denoise-progress'))
correctness['hf_causal_counts'] = dict(correct=sum(sum(s['native_id'] == s['hf_id'] for s in r['comparisons'])
    for r in correctness['hf_causal']['cases']), total=sum(len(r['comparisons']) for r in correctness['hf_causal']['cases']))
for row in correctness['hf_selfspec']['cases']:
    row['passes_20_percent'] = abs(row['relative_tokens_per_round_difference']) <= .2
for row in correctness['hf_denoise']['cases']:
    row['passes_30_percent'] = abs(row['tokens_per_preview_relative_difference']) <= .3

regression = dict(q8=read('q8-final-regressions'), nvfp4=read('nvfp4-final-regressions'), real_mtp=[], real_qwen35=[])
for mode in ('off', 'on'):
    baseline = read('mtp-main-' + mode)
    assert baseline['engine_source']['commit'] == 'f417397f339050e6f400e75db0f1e6acbc40f13a'
    for branch in ('q8', 'nvfp4'):
        current = read('mtp-' + branch + '-' + mode)
        for old, new in zip(baseline['cases'], current['cases'], strict=True):
            tokens_equal = old['token_ids'] == new['token_ids']
            rounds_equal = old['rounds'] == new['rounds']
            regression['real_mtp'].append(dict(branch=branch, mode=mode, prompt=old['prompt'],
                tokens_equal=tokens_equal, rounds_equal=rounds_equal, token_count=len(new['token_ids'])))
            assert tokens_equal and rounds_equal, regression['real_mtp'][-1]
for branch in ('q8', 'nvfp4'):
    baseline, current = read('qwen35-real-nvfp4-main'), read('qwen35-real-nvfp4-' + branch)
    for old, new in zip(baseline['cases'], current['cases'], strict=True):
        equal = old['token_ids'] == new['token_ids'] and old['prompt_ids'] == new['prompt_ids']
        regression['real_qwen35'].append(dict(branch=branch, prompt=old['prompt'], tokens_equal=equal, token_count=len(new['token_ids'])))
        assert equal, regression['real_qwen35'][-1]

quality = {branch: read(branch + '-causal-quality-first50') for branch in ('q8', 'nvfp4')}
for branch, value in quality.items():
    assert len(value['cases']) == 100, branch
historical_path = Path('C:/workspace/runs/T-20261030-clover-nvfp4/quality-native-first50-1024-full-v2/results.json')
historical_raw = historical_path.read_bytes()
historical = json.loads(historical_raw)
assert historical['state'] == 'complete' and historical['protocol']['tokens'] == 1024
old_cases = {(r['dataset'], r['index']): r for r in historical['cases']}
quality['nvfp4']['before_comparison'] = dict(source=str(historical_path), source_sha256=sha(historical_raw),
    same_token_ids=sum(r['token_ids'] == old_cases[r['dataset'], r['index']]['token_ids'] for r in quality['nvfp4']['cases']),
    total=len(quality['nvfp4']['cases']), previous_summary=historical['summary'])
resources = {}
for branch in ('q8', 'nvfp4', 'stock-q8', 'stock-q8-mmap'):
    name = ('stock-q8-none-physical-resource-matrix' if branch == 'stock-q8' else
        'stock-q8-physical-resource-matrix' if branch == 'stock-q8-mmap' else branch + '-physical-resource-matrix')
    matrix = read(name)
    telemetry = read(name, 'physical-gpu-resources.json')
    gaps = [b['t'] - a['t'] for a, b in zip(telemetry['samples'], telemetry['samples'][1:])]
    telemetry['sampling_diagnostics'] = dict(counter_window_seconds=.02,
        actual_gap_median_seconds=statistics.median(gaps), actual_gap_max_seconds=max(gaps),
        note='Counter window differs from actual polling gaps; early stock raw endpoint_resolution_seconds=.02 is a mislabelled counter window, corrected here; byte integrals remain whole-device estimates')
    for interval in telemetry['intervals']:
        interval.pop('endpoint_resolution_seconds', None)
        interval['counter_window_seconds'] = .02
    resources[branch] = dict(benchmark=compact(matrix), telemetry={k: v for k, v in telemetry.items() if k != 'samples'},
        host_memory=sources[name + '/results.json'].get('host_memory'))
long = {label: read(name) for label, name in [('q8', 'q8-long-ttft'), ('nvfp4', 'nvfp4-long-ttft'),
    ('stock_ub256', 'stock-q8-none-long-ttft'), ('stock_mmap', 'stock-q8-long-ttft')]}
long_alternatives = {str(width): optional(f'stock-q8-none-long-ub{width}') for width in (1024, 4096)}
assert long_alternatives['4096']['state'] == 'complete'
long['stock'] = read('stock-q8-none-long-ub4096')
diagnostics = {name: read(name) for name in ('nvfp4-no-ahead-diagnostic', 'nvfp4-old-commit-shape', 'nvfp4-old-shape-reference-pick',
    'nvfp4-old-controls-eager-preview', 'nvfp4-old-controls-eager-both')}
stability = dict(
    q8=output_comparison(matrices['q8_runtime_before'], matrices['q8_p1']),
    nvfp4=output_comparison(matrices['nvfp4_before'], matrices['nvfp4_after']),
    diagnostics={name: output_comparison(matrices['nvfp4_before'], matrix) for name, matrix in diagnostics.items()})
initial_preview = read('q8-initial-preview-hf-numerics')
capture = read('q8-initial-preview-capture')
for row in capture['cases']:
    path = root/'q8-initial-preview-capture'/row['logits_file']
    sources['q8-initial-preview-capture/'+row['logits_file']] = dict(path=str(path),sha256=sha(path.read_bytes()),bytes=path.stat().st_size)
for row in initial_preview['cases']:
    pending=[i for i,t in enumerate(row['block_ids']) if t==151669]
    for side in ('hf','native'):
        confidence=row[side+'_confidence']
        transfer=[i for i in pending if confidence[i]>.95]
        if not transfer:
            transfer=[max(pending,key=lambda i:confidence[i])]
        row[side+'_transfer_rows']=transfer
behavior = {}
for label, filename in [('q8', 'q8-p1-code-json-behavior.json'), ('nvfp4_before', 'nvfp4-before-code-json-behavior.json'),
    ('nvfp4_after', 'nvfp4-final-code-json-behavior.json')]:
    path = root / filename
    raw = path.read_bytes()
    behavior[label] = json.loads(raw)
    sources[filename] = dict(path=str(path), sha256=sha(raw), bytes=len(raw))
failures = []
for path in sorted(root.glob('20*/manifest.json')):
    value = json.loads(path.read_text(encoding='utf-8'))
    invalid_numeric = path.parent.name.endswith(('-official-q8-native-count-p0', '-official-q8-graph-state-audit'))
    if value.get('state') == 'failed' or invalid_numeric:
        failures.append(dict(path=str(path), manifest=value, excluded_from_performance=True,
            invalid_early_q8_numerics=invalid_numeric,
            expected_before_fix_failure='pre-fix-high-slot-reproducer' in path.parent.name))

args.output.mkdir(parents=True, exist_ok=False)
save('provenance', dict(environment=read('machine-and-source-provenance'),
    runtime_inventory=read('runtime-final-inventory'),
    adapter=read('official-q8-adapter-contracts'), pinned_host=read('q8-sized-pinned-bank-probe'),
    canonical_prompts=json.loads(Path('C:/workspace/tasks/T-20261102-clover-q8/short-prompts/manifest.json').read_text(encoding='utf-8'))))
save('matrices', {label: compact(matrix) for label, matrix in matrices.items()})
save('comparison', comparison)
save('loading-ab', dict(primary='Uniform load-mode none for all official comparisons; fixed mmap datasets retained separately',
    preview_mmap=read('stock-q8-preview-perf'), preview_none=stock_preview))
save('correctness', correctness)
save('regression', regression)
save('quality', quality)
save('behavior', behavior)
save('excluded-runs', failures)
save('resources', resources)
save('nvfp4-capacity', dict(causal_5120=optional('nvfp4-cache5120-final-p1'),
    common_4096=matrices['nvfp4_after']['settings']))
save('long-prompts', {label: compact(matrix) for label, matrix in long.items()})
save('long-prefill-ab', long_alternatives)
save('numerical-diagnostics', {label: compact(matrix) for label, matrix in diagnostics.items()})
save('output-stability', stability)
save('hf-initial-preview', initial_preview)
nv_before,nv_after=lookup(matrices['nvfp4_before']),lookup(matrices['nvfp4_after'])
nv_ratios=[row[metric]/nv_before[k][metric] for k,row in nv_after.items() if k[1]!='causal']
save('acceptance',dict(
    T1=all(r['passed'] for r in comparison['T1']),
    T2=all(r['passed'] for r in comparison['T2']),
    T3_conservative=all(r['passed_stricter_any_width'] for r in comparison['T3']),
    T4=all(r['passed_stock'] and r['passed_2x_causal'] for r in comparison['T4']),
    C_preview_state_unchanged=all(r['hashes_before']==r['hashes_after']
        for name in ('q8_graph','nvfp4_graph') for r in correctness[name]['previews']),
    C_graph_vs_eager_qualified_inputs=all(r['logits_bitwise_equal']
        for name in ('q8_graph','nvfp4_graph') for field in ('previews','verifies') for r in correctness[name][field]),
    C_causal_not_below_previous=correctness['hf_causal_counts']['correct']>=155,
    C_selfspec_aggregate=all(r['passes_20_percent'] for r in correctness['hf_selfspec']['cases']),
    C_denoise_aggregate=all(r['passes_30_percent'] for r in correctness['hf_denoise']['cases']),
    C_denoise_failed=[dict(prompt=r['prompt'],width=r['width'],relative_difference=r['tokens_per_preview_relative_difference'])
        for r in correctness['hf_denoise']['cases'] if not r['passes_30_percent']],
    R_with_documented_skips=all(regression[n]['exit_code']==0 for n in ('q8','nvfp4')),
    N_speed_all_25=all(x>1 for x in nv_ratios),N_speed_ratio_min=min(nv_ratios),N_speed_ratio_max=max(nv_ratios),
    N_W32_numeric_reproduction_remaining=[r for r in stability['diagnostics']['nvfp4-old-controls-eager-both'] if not r['same_token_ids']],
    scope='Performance and executed regression gates pass; C denoise progress has four failures; two NVFP4 W32 output differences remain incompletely isolated; never interpret this as all acceptance criteria passed'))
save('outputs', outputs)
# Preserve the controlled old-runtime source and task-local reproduction helpers.
task = Path(__file__).parent
helpers = ['run.py', 'run_regression.py', 'runtime_inventory.py', 'audit_delivery.py', 'observe_memory.py', 'env.ps1', 'machine_snapshot.py', 'check_official_adapter.py',
    'probe_pinned.py', 'reproduce_mmq_before_fix.py', 'check_qwen35_real.py', 'prepare_short_prompts.py',
    'prepare_long_prompts.py', 'run_stock_preview.py', 'bench_gpu_resources.py', 'bench_no_ahead.py',
    'bench_old_commit_shape.py', 'bench_eager_isolation.py', 'bench_cold_experts.py', 'capture_initial_preview.py', 'compare_initial_preview.py',
    'collect_evidence.py', 'render_report.py', 'report-draft.md']
helper_dir = args.output / 'tools'
helper_dir.mkdir()
for name in helpers:
    shutil.copyfile(task / name, helper_dir / name)
patch_dir = args.output / 'patches'
patch_dir.mkdir()
for name, receipt in sources.items():
    if 'run' not in receipt:
        continue
    directory = Path(receipt['run']['output'])
    patch = directory / 'source.patch'
    if patch.exists() and patch.stat().st_size:
        filename = directory.name + '.patch'
        target = patch_dir / filename
        shutil.copyfile(patch, target)
        receipt['source_patch_bundle'] = dict(path='patches/' + filename, sha256=sha(target.read_bytes()))
log_dir = args.output / 'logs'
log_dir.mkdir()
for suffix in ('mmq-large-slot-regression', 'mmq-pre-fix-high-slot-reproducer-expected-failure', 'official-q8-kernels-and-contract-tests'):
    directory = sorted(root.glob('20*-' + suffix))[-1]
    for filename in ('manifest.json', 'stdout.log', 'stderr.log'):
        path = directory / filename
        if path.exists():
            shutil.copyfile(path, log_dir / (directory.name + '-' + filename))
save('sources', sources)
(args.output / 'README.md').write_text((task / 'evidence-readme.md').read_text(encoding='utf-8'), encoding='utf-8')
# Keep recorded raw bytes portable across Git's Windows/Linux newline settings.
(args.output / '.gitattributes').write_bytes(b'# Verbatim evidence: preserve CRLF and whitespace in recorded patches/logs.\n* -text -whitespace\n')
files = {str(path.relative_to(args.output)).replace('\\', '/'): dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    for path in sorted(args.output.rglob('*')) if path.is_file()}
save('inventory', dict(state='complete', files=files, note='Inventory excludes itself; original absolute run paths are provenance locators'))
print(json.dumps({name: all(r.get('passed', r.get('passed_stock', False)) for r in rows) for name, rows in comparison.items()}))
