"""Render the final report from the compact evidence, never from live partial runs."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
def load(name):
    return json.loads((args.evidence / (name + '.json')).read_text(encoding='utf-8'))
def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
        '|' + '|'.join('---' for _ in headers) + '|',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])
def f(value):
    return f'{value:.2f}' if value is not None else '—'
def index(matrix):
    return {(r['prompt'], r['mode'], r['width'] if r['mode'] != 'causal' else 1): r for r in matrix['summary']}
prompts = ('Count', 'Sky', 'Python', 'JSON', 'Chinese')
modes = [('causal', 1), ('selfspec', 8), ('selfspec', 16), ('denoise', 8), ('denoise', 16), ('denoise', 32)]
labels = ['causal', 'selfspec8', 'selfspec16', 'denoise8', 'denoise16', 'denoise32']
matrices, comp = load('matrices'), load('comparison')
sections = []
sections.append('### T1：causal，P0\n\n' + table(['提示', 'FreeToken', 'llama none', 'llama mmap', '相对none'],
    [[r['prompt'], f(r['freetoken']), f(r['stock']), f(r['stock_mmap']), f(r['ratio']) + 'x'] for r in comp['T1']]))
t2 = {(r['prompt'], r['width'], r['rule']): r for r in comp['T2']}
sections.append('### T2：selfspec，P0\n\n每格为FreeToken / stock tok/s。严格两边均驻留；threshold对entropy时FreeToken每次显式清空专家缓存，stock为新CLI进程。Native清空和同步开销保守计入时间。原驻留threshold矩阵另保留。\n\n' +
    table(['提示', '严格W8', '严格W16', 'threshold/entropy W8', 'threshold/entropy W16'],
        [[name, *[f(t2[name, width, rule]['freetoken']) + ' / ' + f(t2[name, width, rule]['stock'])
            for rule, width in [('exact', 8), ('exact', 16), ('threshold', 8), ('threshold', 16)]]] for name in prompts]))
sections.append('### T3：W16去噪，P0\n\n同时给出同宽度比较和更严格的跨W8/W16择快比较；stock均取严格/entropy较快值。最后一列额外取native热/冷较慢值，避免用热native对冷CLI的收益冒充同缓存范围收益。\n\n' +
    table(['提示', 'FreeToken denoise16 热/冷', 'stock最佳W16', 'stock最佳任意宽度', '保守相对任意宽度'],
        [[r['prompt'], f(r['freetoken'])+' / '+f(r['freetoken_cold']), f(r['stock_best_W16']), f(r['stock_best_any_width']), f(r['conservative_ratio']) + 'x'] for r in comp['T3']]))
t4 = {(r['prompt'], r['width']): r for r in comp['T4']}
sections.append('### T4：单次初始块预览，ms\n\n同提示、同anchor、同mask输入；每格FreeToken CUDA event / stock llama_perf decode。\n\n' +
    table(['提示', 'W8', 'W16', 'FreeToken causal step'], [[name,
        f(t4[name, 8]['native_ms']) + ' / ' + f(t4[name, 8]['stock_decode_ms']),
        f(t4[name, 16]['native_ms']) + ' / ' + f(t4[name, 16]['stock_decode_ms']), f(t4[name, 8]['causal_ms'])] for name in prompts]))
long = {label: index(data) for label, data in load('long-prompts').items()}
sections.append('### 1K/4K TTFT，秒\n\n' + table(['提示', 'Q8', 'NVFP4', 'llama none ub4096', 'llama none ub256', 'llama mmap ub256'],
    [[name, *[f(long[label][name, 'causal', 1]['ttft_seconds']) for label in ('q8', 'nvfp4', 'stock', 'stock_ub256', 'stock_mmap')]] for name in ('Long1K', 'Long4K')]))
alternatives = load('long-prefill-ab')
sections.append('主TTFT比较取统一ubatch4096、batch4096；256/1024/4096的完整对照保留：\n\n' +
    table(['stock ubatch', '状态', '1K TTFT秒', '4K TTFT秒'], [[width, v['state'],
        *[f(index(v['data'])[name, 'causal', 1]['ttft_seconds']) if v['state']=='complete' else '失败见原始日志'
          for name in ('Long1K', 'Long4K')]] for width, v in alternatives.items()]) +
    '\n\n这些长提示配置不改变短提示T1～T4的已完成输入、缓存与计时协议；不把ubatch256的结果称为stock最佳prefill性能。')
for label, title in [('q8_p1', '官方Q8六模式矩阵，P1'), ('nvfp4_after', 'NVFP4六模式矩阵，P1')]:
    data = index(matrices[label])
    sections.append('### ' + title + '\n\n' + table(['提示', *labels],
        [[name, *[f(data[name, mode, width]['end_to_end_tokens_per_second']) for mode, width in modes]] for name in prompts]))
for old_label, new_label, title in [('q8_runtime_before', 'q8_p1', 'Q8运行时修复前 → 后'), ('nvfp4_before', 'nvfp4_after', 'NVFP4修复前 → 后')]:
    old, new = index(matrices[old_label]), index(matrices[new_label])
    sections.append('### ' + title + '\n\n每格为前 / 后端到端tok/s，P1。Q8前控制保留正确MMQ和packed dense，仅还原旧运行时。\n\n' +
        table(['提示', *labels], [[name, *[f(old[name, mode, width]['end_to_end_tokens_per_second']) + ' / ' +
            f(new[name, mode, width]['end_to_end_tokens_per_second']) for mode, width in modes]] for name in prompts]))
    ratios=[new[name,mode,width]['end_to_end_tokens_per_second']/old[name,mode,width]['end_to_end_tokens_per_second']
        for name in prompts for mode,width in modes if mode!='causal']
    sections.append('上述25个非causal配置的加速范围：'+f(min(ratios))+'～'+f(max(ratios))+'倍。')
sections.append('### 性能解释\n\n'+
    '运行时前后控制保留同一MoE后端和权重，非causal配置普遍加速；NVFP4 fused MoE CUDA代码未改，支持瓶颈主要在调用、元数据、同步和块间状态处理的判断。'+
    'Q8多行分组MMQ和packed dense解决另一组带宽问题，不能把这两类收益混称为单一kernel加速。单次前向、端到端吞吐和可见token进度分别留证。\n\n'+
    'Q8 P1中Count的D16为70.97 tok/s，高于causal56.85；其余四提示D16仍慢于causal。'+
    '去噪必须反复预测再提交整块，普通文本每轮确定的位置较少，较快的单次前向并不保证较快的生成。'+
    'W32计算量更大，本机五提示矩阵中均未带来优于W16的去噪吞吐。\n\n'+
    'NVFP4每槽1,775,616字节，4096槽载荷6936MiB，覆盖的专家数量更多；Q8每槽3,342,336字节，2560槽8160MiB。'+
    '两者权重精度、输出与缓存覆盖不同，NVFP4绝对速度不能直接作为同权重的llama对比结论。')
stability = load('output-stability')
stability_rows = []
for name in ('q8', 'nvfp4'):
    for group in ('causal/selfspec', 'denoise'):
        rows = [r for r in stability[name] if (r['mode']=='denoise') == (group=='denoise')]
        changed = sorted({(r['prompt'], r['width']) for r in rows if not r['same_token_ids']})
        stability_rows.append([name, group, str(sum(r['same_token_ids'] for r in rows))+'/'+str(len(rows)), str(changed)])
sections.append('### 输出稳定性与NVFP4对照诊断\n\n' +
    table(['分支', '模式', '逐ID一致请求数', '有差异的提示/宽度'], stability_rows) + '\n\n' +
    table(['控制实验', '恢复旧输出数', '仍有差异的提示/宽度'], [[name,
        str(sum(r['same_token_ids'] for r in rows))+'/'+str(len(rows)),
        str(sorted({(r['prompt'],r['width']) for r in rows if not r['same_token_ids']}))]
        for name, rows in stability['diagnostics'].items()]) +
    '\n\nno-ahead恢复逐块单独pending前向；old-commit-shape同时恢复只验证W-1个已提交token；reference-pick进一步恢复Torch惩罚/softmax/argmax。' +
    '这些是因果隔离实验，不混入速度主表；batch形状和置信度浮点归约会改变后续固定顺序，不承诺去噪跨实现逐ID相同。')
remaining=[r for r in stability['diagnostics']['nvfp4-old-controls-eager-both'] if not r['same_token_ids']]
sections.append('恢复独立pending前向使7个变化配置中的5个恢复旧输出，证明这5个与pending合并的数值路径有关。'+
    '其余Python/Chinese W32在恢复旧提交形状、Torch pick、eager预览和eager验证后仍未完全恢复，不能仅归因于CUDA graph或融合pick。\n\n'+
    table(['仍有差异的提示','首次差异token下标（从0）','旧/控制输出长度'],[[r['prompt'],r['first_difference'],str(r['old_tokens'])+'/'+str(r['new_tokens'])] for r in remaining])+
    '\n\n这两例属于尚未完全隔离的数值复现限制；N的25配置速度收益已验证，W32逐ID复现问题仍明确保留，不能宣称所有输出差异都已定位到单独算子。')
correctness = load('correctness')
nv_preview=correctness['nvfp4_preview']['cases']
sections.append('NVFP4五提示单次预览：W8 '+f(min(r['median_ms'] for r in nv_preview if r['width']==8))+'～'+
    f(max(r['median_ms'] for r in nv_preview if r['width']==8))+'ms；W16 '+
    f(min(r['median_ms'] for r in nv_preview if r['width']==16))+'～'+
    f(max(r['median_ms'] for r in nv_preview if r['width']==16))+'ms。此前分支evidence中的eager预览约34ms。')
c = correctness['hf_causal_counts']
sections.append(f"### C：正确性\n\n同权重独立HF强制历史对拍 {c['correct']}/{c['total']}，{100*c['correct']/c['total']:.2f}%；旧分支为155/158。\n\n" +
    '参考保留精确F32反量化Q8 dense并独立模拟Q8_1激活；权重解码与NumPy bitwise核对，不使用服务端GGUF CUDA kernels。\n' +
    '两个causal差异均为HF rank2；HF gap均0.125，其中一个native tie，另一个native gap0.25。\n\n' +
    table(['提示', 'W', 'selfspec平均每轮token相对差异', '±20%'], [[r['prompt'], r['width'], f(100*r['relative_tokens_per_round_difference']) + '%',
        r['passes_20_percent']] for r in correctness['hf_selfspec']['cases']]) + '\n\n' +
    table(['提示', 'W', '去噪平均每轮确定位置相对差异', '±30%'], [[r['prompt'], r['width'], f(100*r['tokens_per_preview_relative_difference']) + '%',
        r['passes_30_percent']] for r in correctness['hf_denoise']['cases']]) +
    '\n\n进度比较统一64-token上限，保留逐轮轨迹；均值容差不等于逐步文本完全一致。两分支预览KV/GDN哈希不变、冷cache重放及每行验证状态审计通过。')
initial = load('hf-initial-preview')
sections.append('相同前缀与相同mask/已固定位置的HF预览对拍：\n\n' +
    table(['提示', 'W', '预览序号', 'argmax一致率', '平均KL', 'HF/native >0.95行数', '跨0.95阈值的行'],
        [[r['prompt'],r['width'],r['step'], f(r['argmax_agreement']*100)+'%', f(r['mean_kl']),
          str(r['hf_above_095'])+'/'+str(r['native_above_095']),str(r['threshold_crossing_rows'])] for r in initial['cases']]) +
    '\n\n该实验固定native输入块，隔离自由生成历史分歧；它不把自由轨迹进度不达标改判为通过。')
sections.append('两边的原始logits统一使用Torch softmax/argmax后，首次预览固定的位置仍不同：\n\n'+
    table(['提示','W','HF固定位置','native固定位置'],[[r['prompt'],r['width'],str(r['hf_transfer_rows']),str(r['native_transfer_rows'])]
        for r in initial['cases'] if r['step']==0])+
    '\n\n因此这4个容差失败已在采样前的模型logits中出现，不能仅归因于融合pick。Count W32有5行跨过0.95阈值，最终token相同但预览次数3对4，离散计数使均值差达到33.33%。'+
    'JSON首次均无位置超过阈值，HF/native选择的最高置信度位置不同，随后排版、内容与自由生成历史分叉。'+
    '独立HF数学参考与服务端量化、BF16 attention/GDN归约路径不同是数值来源；此实验定位到logits/置信度差异，尚未把剩余误差归因到某个单独算子，C去噪进度仍判未达标。')
reg = load('regression')
sections.append('### R：回归\n\n' + table(['分支', 'pytest'], [[name, reg[name]['summary']] for name in ('q8', 'nvfp4')]) +
    '\n\n运行范围是regression evidence命令列出的模型、内核、状态池与调度器回归集合，不是整库所有pytest。\n' +
    '\n\n147跳过项中89是参数布局不适用，46缺完整Qwen4原始权重，4缺专用HF环境，4需vLLM，3需FlashInfer，1缺Gemma样本。\n' +
    '另有真实Flash-Next MTP开/关、main与两feature分支12个配对案例，token和轮次全部一致；真实Qwen3.5族NVFP4三提示在两feature分支也与main逐ID一致。详细输入及保护文件哈希在regression evidence。')
quality = load('quality')
sections.append('### 质量：greedy causal，前50题\n\n统一1024-token上限，关闭思考，checkpoint默认惩罚；这是小样本检查，不是完整榜单。\n\n' +
    table(['分支', 'GSM8K', 'HumanEval', '达到token上限 GSM/HE'], [[name,
        str(quality[name]['summary']['gsm8k']['correct']) + '/50', str(quality[name]['summary']['humaneval']['correct']) + '/50',
        str(quality[name]['summary']['gsm8k']['capped']) + ' / ' + str(quality[name]['summary']['humaneval']['capped'])] for name in ('q8', 'nvfp4')]) +
    '\n\nGSM按最终数值评分；HumanEval使用受限子进程执行题目测试，5秒超时。保留所有输出、失败与评分原因。\n' +
    'Q8五提示P1行为检查中，causal/selfspec Python执行通过，三个去噪宽度失败（shadowed sorted或语法错误）；JSON六模式通过。Python输出带Markdown围栏，执行评分先剥除围栏，不能称作严格遵守only-code格式。')
old_quality = quality['nvfp4']['before_comparison']
sections.append('NVFP4 causal与此前同1024-token协议的100题输出逐ID一致：'+
    str(old_quality['same_token_ids'])+'/'+str(old_quality['total'])+'；旧评分与来源SHA保留在quality evidence。')
behavior = load('behavior')
behavior_rows = []
for branch in ('q8','nvfp4_before','nvfp4_after'):
    for prompt in ('Python','JSON'):
        values={(r['mode'],1 if r['mode']=='causal' else r['width']):r['valid']
            for r in behavior[branch]['cases'] if r['prompt']==prompt}
        behavior_rows.append([branch,prompt,*[values[mode,width] for mode,width in modes]])
sections.append('五提示代码/JSON行为检查，256-token协议：\n\n'+table(['配置','提示',*labels],behavior_rows)+
    '\n\nNVFP4的JSON D8失败在修复前后都存在；三个Python去噪宽度的失败同样存在于旧运行时，不能说是本次加速后才出现。')
resources = load('resources')
sections.append('### 显存与PCIe\n\n下列显存为独立资源矩阵的整卡采样峰值，含桌面；不把带采样的吞吐混入主比较。\n\n' +
    table(['配置', 'NVML整卡峰值 GiB', '实际采样间隔中位数 ms'], [[name,
        f(resources[name]['telemetry']['device_peak_used_bytes']/2**30),
        f(resources[name]['telemetry']['sampling_diagnostics']['actual_gap_median_seconds']*1000)] for name in ('q8', 'nvfp4', 'stock-q8', 'stock-q8-mmap')]) +
    '\n\n逐请求hit率、逻辑H2D字节、物理收发估算和工作集/private峰值在matrices/resources/sources evidence，单位及采样范围分别标注。\n' +
    '旧统计字段cpu_per_layer在GPU offload下沿用miss计数，不能据此断言专家在CPU计算。')
sections.append('同一资源矩阵进程树的CPU内存采样，GiB：\n\n' +
    table(['配置', '峰值工作集RSS', '峰值private', '系统可用最低'], [[name,
        *[f(resources[name]['host_memory'][field]/2**30) for field in ('peak_rss','peak_private','minimum_system_available')]]
        for name in ('q8','nvfp4','stock-q8','stock-q8-mmap')]))
cache_rows = []
for label in ('q8','nvfp4'):
    data = index(resources[label]['benchmark'])
    for mode, width in modes:
        rows = [data[name,mode,width] for name in prompts]
        cache_rows.append([label, mode+str(width) if mode!='causal' else mode,
            f(min(r['expert_cache_hit_rate'] for r in rows)*100)+'–'+f(max(r['expert_cache_hit_rate'] for r in rows)*100)+'%',
            f(min(r['decode_expert_copy_payload_bytes'] for r in rows)/2**30)+'–'+f(max(r['decode_expert_copy_payload_bytes'] for r in rows)/2**30)])
sections.append('五提示资源诊断的缓存命中率与逻辑专家传输范围：\n\n' +
    table(['配置','模式','hit率范围','逻辑H2D GiB/请求范围'],cache_rows) +
    '\n\nstock的ubatch≤8、ubatch>8命中/上传汇总由未修改的日志给出，包含预热，不能直接冒充与native逐请求同口径hit率；完整backend_receipts随matrices提交。')
physical_rows = []
for label in ('q8','nvfp4','stock-q8','stock-q8-mmap'):
    rows = [r for r in resources[label]['telemetry']['intervals'] if not r['warmup']]
    physical_rows.append([label, *[f(min(r[field] for r in rows)/2**30)+'–'+f(max(r[field] for r in rows)/2**30)
        for field in ('rx_estimated_bytes_KB1024','tx_estimated_bytes_KB1024')]])
sections.append('NVML整卡逐请求PCIe采样积分估算范围，GiB；**不是精确硬件字节计数**：\n\n'+
    table(['配置','RX范围','TX范围'],physical_rows))
capacity = load('nvfp4-capacity')['causal_5120']
sections.append('### NVFP4额外容量配置\n\n共同前后比较固定4096槽；额外5120槽、memory_ratio0.95的causal审计状态：'+capacity['state']+'.\n\n'+
    (table(['提示','causal tok/s'],[[r['prompt'],f(r['end_to_end_tokens_per_second'])] for r in capacity['data']['summary']]) if capacity['state']=='complete' else '失败日志保留，不改变4096槽共同预算的主矩阵。')+
    '\n\n更大缓存不是所有提示的通用加速配置；例如已充分命中的Count仍可能变慢，主结论使用固定4096槽。')
sections.append('### 验收状态\n\n' + table(['项', '通过'], [
    ['T1', all(r['passed'] for r in comp['T1'])], ['T2', all(r['passed'] for r in comp['T2'])],
    ['T3 同W16', all(r['passed'] for r in comp['T3'])], ['T3 另取跨宽度最快', all(r['passed_stricter_any_width'] for r in comp['T3'])],
    ['T4', all(r['passed_stock'] and r['passed_2x_causal'] for r in comp['T4'])],
    ['C selfspec进度', all(r['passes_20_percent'] for r in correctness['hf_selfspec']['cases'])],
    ['C 去噪进度', all(r['passes_30_percent'] for r in correctness['hf_denoise']['cases'])]]) +
    '\n\n**C去噪进度有4项未达标，NVFP4有2个W32输出差异未完全隔离；不能将本轮交付写成所有验收条目通过。** '+
    '完整状态见acceptance.json。生成质量失败与输出差异单列，不将速度通过写成所有文本/代码质量通过。')
draft = (Path(__file__).parent / 'report-draft.md').read_text(encoding='utf-8')
assert draft.count('<!-- RESULT_TABLES -->') == 1
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(draft.replace('<!-- RESULT_TABLES -->', '\n\n'.join(sections)), encoding='utf-8')
print(args.output)
