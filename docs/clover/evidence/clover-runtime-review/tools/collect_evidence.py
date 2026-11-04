"""Package review evidence without overwriting the R0 evidence cohort."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination',type=Path,required=True)
a = p.parse_args()
root = Path('C:/workspace/runs/T-20261102-clover-review')
task = Path(__file__).parent
out = a.destination
out.mkdir(parents=True,exist_ok=False)


def copy(source,relative):
    target = out/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)


def classify(name,state):
    if name == 'q8-review-abab-1-after':
        return 'excluded','Development source was extended during this run; replaced by immutable frozen ABAB.'
    if name == 'q8-review-abab-0-before':
        return 'diagnostic','Unpaired run; not a formal before/after result.'
    if name in ('q8-smoke-six-modes','q8-smoke-all-modes'):
        return 'excluded','Early attempt before the corrected causal attention mask; superseded by all-tail audits.'
    if state != 'passed':
        return 'excluded','Failed attempt retained for provenance; not used as performance acceptance.'
    if 'original-abab-v2-' in name:
        return 'legacy-control-repeat','Captured clean R0 source and retained revert snapshot; causal only, separate from R1 frozen full-mode ABAB.'
    if 'frozen-abab-' in name:
        return 'controlled-comparison','Independent immutable imports; two warmups and five measurements per process/case.'
    if 'breakdown' in name or 'block-cost' in name or 'phase-expert' in name:
        return 'instrumented-cost','CUDA call envelopes or optional counter snapshots; separate from the primary uninstrumented matrix.'
    if 'preview-ab-' in name:
        return 'numerical-diagnostic','No speed claim; paired arithmetic and/or identical native clean-state comparison.'
    return 'supplement','See exact command and result scope; not silently merged with the core ABAB cohort.'


registry = []
for directory in sorted(root.iterdir()):
    manifest_path = directory/'manifest.json'
    if not manifest_path.is_file():
        continue
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['state'] == 'running':
        raise RuntimeError(f'Experiment still running: {directory}')
    name = manifest['name']
    category,reason = classify(name,manifest['state'])
    receipt = dict(name=name,local_source=str(directory),category=category,reason=reason,
                   state=manifest['state'],commit=manifest['commit'],started_at_utc=manifest['started_at_utc'],
                   manifest=f'runs/{name}/manifest.json')
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.json','.log','.patch','.py','.ps1','.xml'):
            copy(path,Path('runs')/name/path.relative_to(directory))
    data = root/name
    tensors = []
    if data.is_dir():
        for path in data.rglob('*'):
            if not path.is_file():
                continue
            if path.suffix.lower() in ('.json','.log','.xml'):
                copy(path,Path('runs')/name/'data'/path.relative_to(data))
            elif path.suffix.lower() in ('.pt','.bin'):
                tensors.append(dict(local_path=str(path),bytes=path.stat().st_size,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        receipt['results'] = f'runs/{name}/data/results.json' if (data/'results.json').is_file() else None
    if tensors:
        receipt['large_local_artifacts_not_embedded'] = tensors
    registry.append(receipt)

for name in ('q8-budget-before','q8-budget-after','nvfp4-budget-after'):
    source = root/name/'results.json'
    if source.is_file():
        copy(source,Path('analyses')/(name+'.json'))
analysis_inputs={}
hf=Path('C:/workspace/wt/T-20261102-clover-q8/FreeToken/docs/evidence/clover-runtime/correctness.json')
for derived,native in [('q8-budget-before','q8-breakdown64-before'),('q8-budget-after','q8-breakdown64-after'),
                       ('nvfp4-budget-after','nvfp4-block-costs64')]:
    source=root/native/'results.json'
    if (root/derived/'results.json').is_file():
        analysis_inputs[derived]=dict(native=f'../runs/{native}/data/results.json',
            native_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            hf='../../clover-runtime/correctness.json',hf_sha256=hashlib.sha256(hf.read_bytes()).hexdigest(),
            script='benchmarks/analyze_clover_budget.py',
            reference_scope='Same-Q8 HF arithmetic for Q8; for NVFP4, Q8 progress is only a counterfactual scenario '
                'and not a same-weight oracle. Report uses NVFP4 native progress budgets.')
(out/'analyses').mkdir(exist_ok=True)
(out/'analyses/sources.json').write_text(json.dumps(analysis_inputs,indent=2),encoding='utf-8')
for source in root.glob('*-summary.json'):
    copy(source,Path('analyses')/source.name)
for source in (root/'20261102T101458Z-readonly-audit').glob('*.json'):
    copy(source,Path('audit')/source.name)
for source in root.glob('final-*.json'):
    copy(source,Path('audit')/source.name)
for name in ('mtp-main-off','mtp-main-on','qwen35-real-nvfp4-main'):
    source=Path('C:/workspace/runs/T-20261102-clover-q8')/name/'results.json'
    copy(source,Path('audit/main-controls')/(name+'.json'))
for name in ('run.py','audit.py','run_original_abab.ps1','run_frozen_abab.ps1','run_addons.ps1',
             'run_addons_v2.ps1','render_abab.py','run_final_checks.ps1','check_threshold.py','final_audit.py','collect_evidence.py','campaign-handoff.json','frozen-abab.json',
             'addons.json','final-checks.json'):
    source = task/name
    if source.is_file():
        copy(source,Path('tools')/name)
for source in task.glob('addons-v*.json'):
    copy(source,Path('tools')/source.name)
(out/'registry.json').write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
    experiments=registry),indent=2),encoding='utf-8')
(out/'.gitattributes').write_text('* -text\n',encoding='utf-8')
print(json.dumps(dict(output=str(out),experiments=len(registry))))
