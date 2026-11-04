"""CPU-only software/source inventory; avoid perturbing primary GPU measurements."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
task = Path(__file__).parent
q8 = Path('C:/workspace/wt/T-20261102-clover-q8/FreeToken')
nv4 = Path('C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken')
stock = Path('C:/workspace/wt/T-20261031-clover-gguf/llama.cpp')
build = Path('C:/workspace/builds/T-20261031-clover-gguf/llama-official-cb09-sm120')
def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, encoding='utf-8').strip()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
packages = {}
for name in ('torch', 'triton-windows', 'triton', 'transformers', 'gguf', 'numpy', 'safetensors', 'flashinfer-python', 'pytest'):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
import torch
assert not torch.cuda.is_initialized()
result = dict(state='complete', observed_at=datetime.now(timezone.utc).isoformat(),
    python=sys.version, executable=sys.executable, platform=platform.platform(), packages=packages,
    torch_cuda_runtime=torch.version.cuda, cuda_initialized=False,
    official_weight_provenance=json.loads(Path('C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/provenance.json').read_text(encoding='utf-8')),
    design_sha256=digest(Path('C:/Users/PC/Desktop/方案设计_Clover官方Q8_GGUF适配与块去噪性能修复.md')),
    sources={name: dict(path=str(path), commit=git(path, 'rev-parse', 'HEAD'), dirty=git(path, 'status', '--porcelain'))
        for name, path in [('q8', q8), ('nvfp4', nv4), ('stock', stock)]},
    baseline_binaries={p.name: dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)
        for p in (build / 'bin').iterdir() if p.name in ('llama-server.exe', 'llama-speculative-simple.exe', 'llama.dll', 'ggml-cuda.dll')},
    cmake_flags=[line for line in (build / 'CMakeCache.txt').read_text(encoding='utf-8').splitlines()
        if any(line.startswith(key + ':') for key in ('CMAKE_CUDA_ARCHITECTURES', 'CMAKE_CUDA_COMPILER', 'CMAKE_BUILD_TYPE',
            'GGML_CUDA', 'GGML_CUDA_FA_ALL_QUANTS', 'LLAMA_CURL', 'LLAMA_BUILD_TESTS', 'LLAMA_BUILD_SERVER'))],
    nvfp4_cuda_kernel_changes=git(nv4, 'diff', '--name-only', '2fd00ce', 'HEAD', '--', 'python/freetoken/kernel/csrc'))
assert result['sources']['stock']['commit'] == 'cb09d4355d8bf968bb907c5de03cf42d8aa073ce'
assert result['sources']['stock']['dirty'] == ''
assert result['nvfp4_cuda_kernel_changes'] == ''
(args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(packages=packages, torch_cuda_runtime=torch.version.cuda, stock_clean=True, nvfp4_cuda_unchanged=True)))
