param([string]$WaitManifest)
$ErrorActionPreference='Stop'
. C:/workspace/tasks/T-20261102-clover-q8/env.ps1
$py='C:/workspace/envs/T-20261102-clover-q8/runtime/Scripts/python.exe'
$runner='C:/workspace/tasks/T-20261102-clover-review/run.py'
$q8Before='C:/workspace/wt/T-20261102-clover-review/FreeToken-q8-before'
$nvBefore='C:/workspace/wt/T-20261102-clover-review/FreeToken-nvfp4-before'
$q8After='C:/workspace/wt/T-20261102-clover-review/FreeToken-q8-reviewed'
$nvAfter='C:/workspace/wt/T-20261102-clover-review/FreeToken-nvfp4-reviewed'
$model='C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf'
$nvModel='C:/workspace/models/bundles/clover-nvfp4/4b0402b6-routed-v1'
$root='C:/workspace/runs/T-20261102-clover-review'
if($WaitManifest){
 while(($s=Get-Content -LiteralPath $WaitManifest -Raw|ConvertFrom-Json).state -eq 'running'){Start-Sleep -Seconds 2}
 if($s.state -ne 'passed'){throw 'Preceding GPU job failed'}
}
$queue=[ordered]@{state='running';jobs=@()}
function Save-Queue{$queue|ConvertTo-Json -Depth 8|Set-Content C:/workspace/tasks/T-20261102-clover-review/frozen-abab.json -Encoding utf8}
try{
 foreach($precision in @('q8','nvfp4')){
  $checkpoint=if($precision -eq 'q8'){$model}else{$nvModel}
  $slots=if($precision -eq 'q8'){'2560'}else{'4096'}
  foreach($i in 0..3){
   $label=if($i%2 -eq 0){'before'}else{'after'}
   $source=if($precision -eq 'q8'){if($label -eq 'before'){$q8Before}else{$q8After}}else{if($label -eq 'before'){$nvBefore}else{$nvAfter}}
   $name="$precision-frozen-abab-$i-$label"
   $env:PYTHONPATH="$source/python;$source"
   $env:TORCH_EXTENSIONS_DIR="C:/workspace/builds/T-20261102-clover-review/extensions-$precision-$label"
   $queue.jobs += [ordered]@{name=$name;source=$source;state='running'}
   Save-Queue
   & $py $runner --name $name --cwd $source --env "PYTHONPATH=$env:PYTHONPATH" --env "TORCH_EXTENSIONS_DIR=$env:TORCH_EXTENSIONS_DIR" -- $py "$q8After/benchmarks/bench_clover.py" --model $checkpoint --output "$root/$name" --cache-slots $slots --memory-ratio .98 --tokens 256 --warmups 2 --repeats 5 --modes causal,selfspec:8,selfspec:16,denoise:8,denoise:16,denoise:32
   if($LASTEXITCODE -ne 0){$queue.jobs[-1].state='failed';throw "$name failed"}
   $queue.jobs[-1].state='passed'
   Save-Queue
  }
 }
 $queue.state='passed'
}catch{$queue.state='failed';$queue.error=$_.ToString();throw}finally{Save-Queue}
