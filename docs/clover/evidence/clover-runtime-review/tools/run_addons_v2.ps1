$ErrorActionPreference='Stop'
. C:/workspace/tasks/T-20261102-clover-q8/env.ps1
$py='C:/workspace/envs/T-20261102-clover-q8/runtime/Scripts/python.exe'
$runner='C:/workspace/tasks/T-20261102-clover-review/run.py'
$q8='C:/workspace/wt/T-20261102-clover-q8/FreeToken'
$nv4='C:/workspace/wt/T-20261030-clover-nvfp4/FreeToken'
$model='C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf'
$nvModel='C:/workspace/models/bundles/clover-nvfp4/4b0402b6-routed-v1'
$root='C:/workspace/runs/T-20261102-clover-review'
$env:CLOVER_HF_SOURCE='C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff'
$env:CLOVER_GGUF_CHECKPOINT=$model
while(($prior=Get-Content C:/workspace/tasks/T-20261102-clover-review/frozen-abab.json -Raw|ConvertFrom-Json).state -eq 'running'){Start-Sleep -Seconds 2}
if($prior.state -ne 'passed'){throw 'Frozen ABAB prerequisite failed'}
$queue=[ordered]@{state='running';jobs=@()}
function Save-Queue{$queue|ConvertTo-Json -Depth 8|Set-Content C:/workspace/tasks/T-20261102-clover-review/addons.json -Encoding utf8}
function Run-Job([string]$Name,[string]$Source,[string[]]$Arguments){
 $env:PYTHONPATH="$Source/python;$Source"
 $queue.jobs += [ordered]@{name=$Name;state='running'};Save-Queue
 & $py $runner --name $Name --cwd $Source --env "PYTHONPATH=$env:PYTHONPATH" --env "CLOVER_HF_SOURCE=$env:CLOVER_HF_SOURCE" --env "CLOVER_GGUF_CHECKPOINT=$env:CLOVER_GGUF_CHECKPOINT" -- $py @Arguments
 if($LASTEXITCODE -ne 0){$queue.jobs[-1].state='failed';throw "$Name failed"}
 $queue.jobs[-1].state='passed';Save-Queue
}
try{
 Run-Job 'optional-kernel-and-mapping-tests-v2' $q8 @('-m','pytest','-q','-ra','tests/models/test_clover_generation.py','tests/kernels/test_clover_preview.py','tests/models/test_clover_llama_metrics.py')
 Run-Job 'q8-paired-split-kv-v2' $q8 @("$q8/benchmarks/bench_clover_split_kv.py",'--model',$model,'--output',"$root/q8-paired-split-kv-v2")
 Run-Job 'q8-split-graph-audit' $q8 @("$q8/benchmarks/check_clover_graphs.py",'--model',$model,'--output',"$root/q8-split-graph-audit",'--cache-slots','2560','--memory-ratio','.98','--split-kv')
 foreach($hostMode in @('off','on')){
  $flags=if($hostMode -eq 'on'){@('--embedding-host')}else{@()}
  Run-Job "q8-host-embedding-$hostMode" $q8 (@("$q8/benchmarks/bench_clover.py",'--model',$model,'--output',"$root/q8-host-embedding-$hostMode",'--cache-auto','--memory-ratio','.9','--tokens','128','--warmups','2','--repeats','5','--modes','causal')+$flags)
 }
 Run-Job 'q8-host-w32-capacity' $q8 @("$q8/benchmarks/bench_clover.py",'--model',$model,'--output',"$root/q8-host-w32-capacity",'--cache-auto','--embedding-host','--memory-ratio','.9','--tokens','64','--prompts','Count','--warmups','1','--repeats','1','--modes','denoise:32')
 Run-Job 'q8-phase-expert-traffic' $q8 @("$q8/benchmarks/bench_clover.py",'--model',$model,'--output',"$root/q8-phase-expert-traffic",'--cache-slots','2560','--memory-ratio','.98','--tokens','64','--warmups','2','--repeats','3','--modes','denoise:16','--runtime-breakdown','--phase-cache-stats')
 Run-Job 'nvfp4-block-costs64' $nv4 @("$q8/benchmarks/bench_clover.py",'--model',$nvModel,'--output',"$root/nvfp4-block-costs64",'--cache-slots','4096','--memory-ratio','.98','--tokens','64','--warmups','2','--repeats','5','--modes','causal,denoise:16','--runtime-breakdown')
 foreach($source in @($q8,$nv4)){
  $label=if($source -eq $q8){'q8'}else{'nvfp4'}
  Run-Job "$label-review-regressions" $source @('C:/workspace/tasks/T-20261102-clover-q8/run_regression.py','--output',"$root/$label-review-regressions")
  foreach($mode in @('off','on')){
   Run-Job "mtp-$label-review-$mode" $source @("$source/benchmarks/check_qwen4_mtp.py",'--model','C:/workspace/models/bundles/flash-next/mtpdecode2-20261026/checkpoints/flash-next-strata-q2','--mtp-runtime','C:/workspace/models/bundles/flash-next/mtpdecode2-20261026/checkpoints/flash-next-mtp','--code-root',$source,'--mode',$mode,'--output',"$root/mtp-$label-review-$mode")
  }
  Run-Job "qwen35-$label-review" $source @('C:/workspace/tasks/T-20261102-clover-q8/check_qwen35_real.py','--code-root',$source,'--model','C:/workspace/models/inputs/qwen3.6-35b-a3b-nvfp4/1355db6a052410cfd62085d94b58866fd0f2c3c5','--output',"$root/qwen35-$label-review")
 }
 $queue.state='passed'
}catch{$queue.state='failed';$queue.error=$_.ToString();throw}finally{Save-Queue}

