"""Render current tables from retained ABAB summaries, keeping raw scopes separate."""
import json
from pathlib import Path

root=Path('C:/workspace/runs/T-20261102-clover-review')
repo=Path('C:/workspace/wt/T-20261102-clover-q8/FreeToken')
rows=[]
for label in ('q8','nvfp4'):
 d=json.loads((root/f'{label}-frozen-summary.json').read_text(encoding='utf-8'))
 cases={(r['prompt'],r['mode'],r['width']):r for r in d['cases']}
 assert all(r['all_measured_output_ids_equal'] for r in cases.values())
 rows += [f'### {label.upper()} 核心增量复测', '',
 '表中吞吐为两个 after 进程共 10 次实测的合并中位数；TTFT 单位秒。', '',
 '| 提示 | causal 端到端 | causal 纯 decode | causal TTFT | ss8 | ss16 | d8 | d16 | d32 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for prompt in ('Count','Sky','Python','JSON','Chinese'):
  causal=cases[prompt,'causal',8]['metrics']
  values=[causal['end_to_end_tokens_per_second']['after_median'],
          causal['decode_tokens_per_second']['after_median']]
  ttft=causal['ttft_seconds']['after_median']
  modes=[('selfspec',8),('selfspec',16),('denoise',8),('denoise',16),('denoise',32)]
  tail=[cases[prompt,mode,width]['metrics']['end_to_end_tokens_per_second']['after_median'] for mode,width in modes]
  rows.append(f'| {prompt} | {values[0]:.1f} | {values[1]:.1f} | {ttft:.3f} | '+
              ' | '.join(f'{v:.1f}' for v in tail)+' |')
 rows += ['', '| 模式 | 第一组 after/before 范围 | 第二组 after/before 范围 |', '|---|---:|---:|']
 for mode in ('causal','selfspec','denoise'):
  pairs=[r['metrics']['end_to_end_tokens_per_second']['after_over_before_pairs']
         for r in cases.values() if r['mode']==mode]
  spans=[f'{min(p[i] for p in pairs):.3f}–{max(p[i] for p in pairs):.3f}' for i in range(2)]
  rows.append(f'| {mode} | {spans[0]} | {spans[1]} |')
 rows += ['', '范围跨提示/宽度，是观测范围，不是置信区间。每进程中位数和五次样本范围均在 '+
          f'[机器可读汇总](evidence/clover-runtime-review/analyses/{label}-frozen-summary.json) 中。', '']
rows += ['两种精度各 30 个配置，20 次实测（每侧 10 次）的输出 ID 在各配置内全部相同。',
 '这只证明此样本集的增量稳定性，不会消除 R0 的 HF 差异，也不意味着 selfspec 与 causal 相同。',
 '本轮 denoise 热态吞吐基本持平；收益主要来自 causal/selfspec 的惩罚与验证处理。',
 'Q8 的部分 before 进程仍有时间漂移，所以同时保留两组成对比值，不把最大的单项比值当作稳定收益。',
 'NVFP4 的 Sky selfspec16 在两个 after 进程中位数为 65.9/58.5 tok/s，第二组成对为 -1.5%；',
 'Sky denoise8 第二组为 -3.6%。这些单项仍有运行间波动，本轮不宣称全模式稳定提速，也不把波动指定归因于 allocator。']
path=repo/'docs/clover-runtime-review.md'
text=path.read_text(encoding='utf-8')
assert '<!-- R1_ABAB_TABLES -->' in text
path.write_text(text.replace('<!-- R1_ABAB_TABLES -->','\n'.join(rows)),encoding='utf-8')
print(json.dumps(dict(state='complete',tables=4)))
