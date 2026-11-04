"""Validate and summarize the final, deliberately small decode snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--label-prefix', default='final-decode-')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path('C:/workspace/runs/T-20261104-clover-improve')
# The harness carries the default request width 8 even for causal; causal state
# reservation is independently zero because its mode is excluded from the maximum.
expected = [('causal', 8), ('selfspec', 8), ('selfspec', 16), ('denoise', 16), ('denoise', 32)]
report = dict(state='running', scope='One warmup and one measured request per mode/input; '
    'EOS respected. Decode excludes the first visible burst and terminal EOS. '
    'This snapshot is not an ABAB claim or a statistical speedup estimate.', inputs=[], rows=[])
commit = None
prompt_ids = {}
causal_outputs = {}

for mode, width in expected:
    label = a.label_prefix + (mode if mode == 'causal' else f'{mode}-{width}')
    manifests = [path for path in root.glob('*-' + label + '/manifest.json')
        if json.loads(path.read_bytes())['state'] == 'passed']
    assert len(manifests) == 1, (label, manifests)
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_bytes())
    folder = manifest_path.parent / 'data'
    result_path = folder / 'results.json'
    result = json.loads(result_path.read_bytes())
    scope = json.loads((folder / 'measurement-scope.json').read_bytes())
    assert result['state'] == 'complete' and not manifest['dirty'], manifest
    assert scope['commit'] == manifest['commit']
    commit = commit or manifest['commit']
    assert manifest['commit'] == commit
    settings = result['settings']
    for key, value in dict(warmups=1, repeats=1, context=4608, token_budget=256,
            cache_auto=True, memory_ratio=.9, accept_rule='exact', embedding_host=True,
            prefix_cache_reuse=False, penalties='checkpoint defaults').items():
        assert settings[key] == value, (key, settings[key], value)
    assert len(result['runs']) == 4
    report['inputs'].append(dict(label=label, manifest=manifest,
        result=str(result_path), sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        scope=scope, settings=settings))
    row = dict(mode=mode, width=width, cache_slots=settings['cache_slots'], measurements={})
    for name in ('FinalShort', 'Final4K'):
        ids = result['prompts'][name]['token_ids']
        assert len(ids) == 4096 if name == 'Final4K' else 0 < len(ids) < 4096
        if name in prompt_ids:
            assert prompt_ids[name] == ids
        prompt_ids[name] = ids
        samples = [item for item in result['runs'] if item['prompt'] == name]
        assert len(samples) == 2 and sum(item['warmup'] for item in samples) == 1
        sample = next(item for item in samples if not item['warmup'])
        assert (sample['mode'], sample['width']) == (mode, width)
        count = sample['output_tokens']
        assert count == len(sample['token_ids']) and 0 < count <= 256
        events = sample['stream_event_arrival_seconds']
        assert len(events) in (count, count + 1)
        visible = events[:count]
        assert all(b >= a for a, b in zip(visible, visible[1:]))
        first = sum(stamp == visible[0] for stamp in visible)
        elapsed = visible[-1] - visible[0]
        decode = (count - first) / elapsed if elapsed > 0 else None
        assert decode == sample['decode_tokens_per_second']
        if mode == 'causal':
            causal_outputs[name] = sample['token_ids']
        row['measurements'][name] = dict(input_tokens=len(ids), output_tokens=count,
            decode_tokens_per_second=decode, decode_tokens=count-first,
            decode_seconds=elapsed, first_burst_tokens=first,
            ttft_seconds=sample['ttft_seconds'], wall_seconds=sample['wall_seconds'],
            end_to_end_tokens_per_second=sample['end_to_end_tokens_per_second'],
            tokens_per_round=sample['tokens_per_round'], tokens_per_preview=sample['tokens_per_preview'],
            output_equal_to_causal=sample['token_ids']==causal_outputs[name])
    report['rows'].append(row)

report.update(state='complete', commit=commit, prompts=prompt_ids)
(a.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
lines = [f'实测 commit：`{commit}`。', '',
    '| Decode 模式 | 短输入 TPS | 短输入输出 tokens | 4K 输入 TPS | 4K 输入输出 tokens | 专家槽数 |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
labels = {'causal': '自回归', 'selfspec': '自投机', 'denoise': '块扩散'}
def rate(value):
    return f'{value:.2f}' if value is not None else 'N/A（单批输出）'
for row in report['rows']:
    short, long = (row['measurements'][name] for name in ('FinalShort','Final4K'))
    label = labels[row['mode']] + ('' if row['mode']=='causal' else f" W{row['width']}")
    lines.append(f"| {label} | {rate(short['decode_tokens_per_second'])} | {short['output_tokens']} | "
        f"{rate(long['decode_tokens_per_second'])} | {long['output_tokens']} | {row['cache_slots']} |")
lines += ['', f"短输入为 {len(prompt_ids['FinalShort'])} tokens，长输入为 4096 tokens；输出上限 256。",
    '每项预热一次、正式测一次；使用模型默认惩罚、自动缓存 .9、host embedding，按各模式声明宽度预留状态。',
    'Decode TPS 从首批可见输出之后计时，扣除首批 tokens 与终止 EOS；各模式输出内容可能不同。']
(a.output / 'table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(dict(commit=commit, rows=report['rows']), ensure_ascii=False))
