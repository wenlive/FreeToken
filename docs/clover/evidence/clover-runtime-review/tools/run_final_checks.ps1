$ErrorActionPreference='Stop'
. C:/workspace/tasks/T-20261102-clover-q8/env.ps1
$py='C:/workspace/envs/T-20261102-clover-q8/runtime/Scripts/python.exe'
$runner='C:/workspace/tasks/T-20261102-clover-review/run.py'
$q8='C:/workspace/wt/T-20261102-clover-q8/FreeToken'
$nv4='C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken'
$q8Before='C:/workspace/wt/T-20261102-clover-review/FreeToken-q8-before'
$nvBefore='C:/workspace/wt/T-20261102-clover-review/FreeToken-nvfp4-before'
$model='C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf'
$nvModel='C:/workspace/models/bundles/clover-nvfp4/4b0402b6-routed-v1'
$root='C:/workspace/runs/T-20261102-clover-review'
$env:CLOVER_HF_SOURCE='C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff'
$env:CLOVER_GGUF_CHECKPOINT=$model
if((Get-Content C:/workspace/tasks/T-20261102-clover-review/addons.json -Raw|ConvertFrom-Json).state -ne 'passed'){throw 'Add-on prerequisite failed'}
$queue=[ordered]@{state='running';jobs=@()}
function Save-Queue{$queue|ConvertTo-Json -Depth 8|Set-Content C:/workspace/tasks/T-20261102-clover-review/final-checks.json -Encoding utf8}
function Run-Job([string]$Name,[string]$Source,[string[]]$Arguments){
 $env:PYTHONPATH="$Source/python;$Source"
 $queue.jobs += [ordered]@{name=$Name;state='running'};Save-Queue
 & $py $runner --name $Name --cwd $Source --env "PYTHONPATH=$env:PYTHONPATH" --env "CLOVER_HF_SOURCE=$env:CLOVER_HF_SOURCE" --env "CLOVER_GGUF_CHECKPOINT=$env:CLOVER_GGUF_CHECKPOINT" -- $py @Arguments
 if($LASTEXITCODE -ne 0){$queue.jobs[-1].state='failed';throw "$Name failed"}
 $queue.jobs[-1].state='passed';Save-Queue
}
try{
 Run-Job 'nvfp4-budget-summary-tests' $nv4 @('-m','pytest','-q','tests/models/test_clover_budget.py')
 Run-Job 'nvfp4-split-graph-audit' $nv4 @("$nv4/benchmarks/check_clover_graphs.py",'--model',$nvModel,'--output',"$root/nvfp4-split-graph-audit",'--cache-slots','4096','--memory-ratio','.98','--split-kv')
 foreach($hostMode in @('off','on')){
  $flags=@();if($hostMode -eq 'on'){$flags=@('--embedding-host')}
  Run-Job "q8-auto-w32-resource-$hostMode" $q8 (@("$q8/benchmarks/bench_clover_gpu_resources.py",'--resource-frontend','native','--model',$model,'--output',"$root/q8-auto-w32-resource-$hostMode",'--cache-auto','--memory-ratio','.9','--tokens','64','--prompts','Count','--warmups','1','--repeats','2','--modes','denoise:32')+$flags)
 }
 foreach($label in @('q8','nvfp4')){
  $weight=if($label -eq 'q8'){$model}else{$nvModel}
  $slots=if($label -eq 'q8'){'2560'}else{'4096'}
  foreach($side in @('before','after')){
   if($label -eq 'q8'){$source=if($side -eq 'before'){$q8Before}else{$q8}}
   else{$source=if($side -eq 'before'){$nvBefore}else{$nv4}}
   $name="$label-threshold-correctness-$side"
   Run-Job $name $source @("$q8/benchmarks/bench_clover.py",'--model',$weight,'--output',"$root/$name",'--cache-slots',$slots,'--memory-ratio','.98','--tokens','64','--prompts','Count,JSON','--warmups','2','--repeats','3','--modes','selfspec:8,selfspec:16','--accept-rule','threshold')
  }
 }
 $queue.state='passed'
}catch{$queue.state='failed';$queue.error=$_.ToString();throw}finally{Save-Queue}
