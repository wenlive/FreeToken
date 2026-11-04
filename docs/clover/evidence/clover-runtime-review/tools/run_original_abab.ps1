$ErrorActionPreference = 'Stop'
. C:/workspace/tasks/T-20261102-clover-q8/env.ps1
$py = 'C:/workspace/envs/T-20261102-clover-q8/runtime/Scripts/python.exe'
$runner = 'C:/workspace/tasks/T-20261102-clover-review/run.py'
$model = 'C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf'
foreach ($turn in 0..3) {
    $label = if ($turn % 2 -eq 0) { 'old' } else { 'delivered' }
    $source = if ($label -eq 'old') { 'C:/workspace/wt/T-20261102-clover-q8/FreeToken-q8-runtime-before' } else { 'C:/workspace/wt/T-20261102-clover-review/FreeToken-q8-before' }
    $env:PYTHONPATH = "$source/python;$source"
    $result = "C:/workspace/runs/T-20261102-clover-review/original-abab-v2-$turn-$label"
    & $py $runner --name "original-abab-v2-$turn-$label" --cwd $source --env "PYTHONPATH=$env:PYTHONPATH" -- $py "$source/benchmarks/bench_clover.py" --model $model --output $result --cache-slots 2560 --memory-ratio .95 --modes causal --tokens 256 --warmups 2 --repeats 5
    if ($LASTEXITCODE -ne 0) { throw "Original ABAB failed at $turn" }
}
