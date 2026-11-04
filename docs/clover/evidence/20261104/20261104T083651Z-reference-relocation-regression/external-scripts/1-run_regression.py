"""Run applicable regression files, including official local Clover weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
files = ['tests/models/qwen4_exp', 'tests/models/test_qwen3_5_moe_weight.py',
    'tests/models/test_gemma4_gguf_rope.py', 'tests/kernels/test_gemma4_fused_ops.py',
    'tests/moe/test_nvfp4_backends.py', 'tests/moe/test_fused_copy.py',
    'tests/kvcache/test_mtp_pool.py', 'tests/kvcache/test_linear_state_pool_alloc.py',
    'tests/models/test_clover_model.py', 'tests/models/test_clover_generation.py',
    'tests/models/test_clover_offload.py', 'tests/models/test_clover_gguf.py',
    'tests/models/test_clover_eval.py', 'tests/models/test_clover_llama_metrics.py',
    'benchmarks/clover/tests/test_clover_eval.py', 'benchmarks/clover/tests/test_clover_llama_metrics.py',
    'tests/scheduler/test_clover_prefill_bundle.py', 'tests/kernels/test_clover_preview.py',
    'tests/kernels/test_clover_gguf.py', 'tests/models/test_clover_real_layers.py',
    'tests/models/test_clover_bf16_blocks.py']
selected = [f for f in files if Path(f).exists()]
started = time.monotonic()
command = [sys.executable, '-m', 'pytest', '-q', '-ra', '--junitxml', str(args.output / 'junit.xml'), *selected]
process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
print(process.stdout, flush=True)
(args.output / 'pytest.log').write_text(process.stdout, encoding='utf-8')
result = dict(state='complete' if not process.returncode else 'failed', command=command,
    absent_files=[f for f in files if f not in selected], exit_code=process.returncode,
    seconds=time.monotonic() - started, summary=process.stdout.splitlines()[-1],
    weight_environment={k: os.environ.get(k) for k in ('CLOVER_HF_SOURCE', 'CLOVER_GGUF_CHECKPOINT', 'FREETOKEN_QWEN4EXP_MODEL')})
(args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
sys.exit(process.returncode)
