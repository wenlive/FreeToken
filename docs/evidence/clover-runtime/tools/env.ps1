. C:/workspace/tasks/SP-20261026/Enter-BuildEnvironment.ps1 -VisualStudio 2022 -Environment runtime
$env:PATH = 'C:/workspace/envs/T-20261102-clover-q8/runtime/Scripts;' + $env:PATH
$env:PYTHONPATH = 'C:/workspace/wt/T-20261102-clover-q8/FreeToken/python;C:/workspace/wt/T-20261102-clover-q8/FreeToken'
$env:FREETOKEN_KERNEL_CACHE_DIR = 'C:/workspace/builds/T-20261102-clover-q8/kernel-cache'
$env:TRITON_CACHE_DIR = 'C:/workspace/cache/triton/clover-q8'
$env:TORCH_EXTENSIONS_DIR = 'C:/workspace/builds/T-20261102-clover-q8/torch-extensions'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
