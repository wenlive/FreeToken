"""Actual streamed dense generation, adapter controls and fresh-process reload."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--adapter', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--minimum-canary-accuracy', type=float)
    parser.add_argument('--check-cache-rebuild', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.minimum_canary_accuracy is not None and (not args.corpus or not 0 <= args.minimum_canary_accuracy <= 1):
        parser.error('minimum-canary-accuracy requires corpus and a value in [0,1]')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from freetoken.core import SamplingParams
    from freetoken.layers.base import BaseOP
    from freetoken.llm import LLM
    from freetoken.lora.ops import PackedLinear
    from freetoken.mm.config import MultimodalConfig

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = dict(moe_strategy='fused', dense_layer_streaming=True, max_running_req=args.batch_size,
                    memory_ratio=.70, num_token_override=2048, max_seq_len_override=2048,
                    max_extend_tokens=512, cuda_graph_max_bs=0, cache_type='naive',
                    lora_path=str(args.adapter))
    result = {'status': 'RUNNING', 'scope': 'Complete 27B FreeToken streamed inference using the supplied publication',
              'model': str(args.model), 'adapter': str(args.adapter), 'settings': settings,
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        (args.output / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    llm = None
    try:
        save()
        started = time.perf_counter()
        with torch.inference_mode():
            llm = LLM(str(args.model), mm=MultimodalConfig(disabled_encoders=frozenset({'vision', 'audio'})), **settings)
        result['load_seconds'] = time.perf_counter() - started
        result['binding'] = llm.engine.lora_info
        manager = llm.engine.model._dense_streamer
        before_hashes = manager.hashes()
        result['streaming'] = manager.inventory()
        result['host_allocator_after_load'] = torch.cuda.memory.host_memory_stats()
        save()
        print('Real dense engine ready; testing generation and adapter controls', flush=True)
        ops, seen = [], set()
        def walk(op):
            if id(op) in seen:
                return
            seen.add(id(op))
            if getattr(op, '_lora_linear', None) is not None:
                ops.append(op)
            for value in vars(op).values():
                if isinstance(value, BaseOP):
                    walk(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, BaseOP):
                            walk(item)
        walk(llm.engine.model)
        @contextmanager
        def factors(mode):
            original = [op._lora_linear for op in ops]
            try:
                for op, packed in zip(ops, original):
                    if mode.startswith('disabled'):
                        op._lora_linear = None
                    elif mode == 'zero_B':
                        op._lora_linear = PackedLinear(packed.a, torch.zeros_like(packed.b), packed.segments)
                yield
            finally:
                for op, packed in zip(ops, original):
                    op._lora_linear = packed
        logits, capture_logits = [], True
        original_forward = llm.engine.model.forward
        def forward():
            output = original_forward()
            if not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite real dense inference logits')
            if capture_logits:
                logits.append(output[0].detach().cpu().contiguous())
            return output
        llm.engine.model.forward = forward
        cases = [('What is 1+1? Answer with one digit only.', '2'),
                 ('中国的首都是哪里？只回答城市名称。', '北京'),
                 ('Translate 你好 into English. Give only the translation.', 'Hello')]
        outputs = []
        with torch.inference_mode():
            for question, expected in cases:
                prompt = llm.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
                captured = {}
                for mode in ['disabled', 'zero_B', 'enabled', 'enabled_repeat', 'disabled_repeat']:
                    logits.clear()
                    started = time.perf_counter()
                    with factors(mode):
                        answer = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8))[0]
                    torch.cuda.synchronize()
                    captured[mode] = {'answer': answer, 'first_logits': logits[0].clone(),
                                      'wall_seconds': time.perf_counter() - started, 'forward_calls': len(logits)}
                    print(f'{mode}: {answer["text"]!r}', flush=True)
                def same(a, b):
                    return torch.equal(captured[a]['first_logits'].view(torch.uint8), captured[b]['first_logits'].view(torch.uint8))
                delta = captured['enabled']['first_logits'].float() - captured['disabled']['first_logits'].float()
                checks = {'zero_B_logits_E2': same('zero_B', 'disabled'),
                          'repeat_enabled_logits_E2': same('enabled', 'enabled_repeat'),
                          'repeat_disabled_logits_E2': same('disabled', 'disabled_repeat'),
                          'adapter_logits_effect_nonzero': bool(delta.count_nonzero()),
                          'base_semantic_sanity': expected.lower() in captured['disabled']['answer']['text'].lower()}
                outputs.append({'question': question, 'prompt_tokens': prompt, 'checks': checks,
                    'adapter_max_abs_logit_effect': float(delta.abs().max()),
                    'adapter_relative_logit_L2': float(delta.norm() / captured['disabled']['first_logits'].float().norm()),
                    'modes': {mode: {key: value for key, value in row.items() if key != 'first_logits'} for mode, row in captured.items()}})
                torch.save({mode: row['first_logits'] for mode, row in captured.items()}, args.output / f'logits-{len(outputs)-1}.pt')
                result['outputs'] = outputs
                save()
                assert all(checks.values()), checks
            if args.corpus:
                corpus = json.loads((args.corpus / 'manifest.json').read_text(encoding='utf-8'))
                artifact = json.loads((args.adapter / 'manifest.json').read_text(encoding='utf-8'))
                assert corpus['base'] == artifact['base'], 'Quality corpus belongs to another base'
                for name, digest in corpus['files'].items():
                    assert hashlib.sha256((args.corpus / name).read_bytes()).hexdigest() == digest
                result['quality'] = {}
                result['corpus_manifest_sha256'] = hashlib.sha256((args.corpus / 'manifest.json').read_bytes()).hexdigest()
                capture_logits = False
                logits.clear()
                for mode in ['disabled', 'enabled']:
                    result['quality'][mode] = {}
                    for name in ['canary-holdout.jsonl', 'replay.jsonl']:
                        rows = [json.loads(line) for line in (args.corpus / name).read_text(encoding='utf-8').splitlines() if line.strip()]
                        records = []
                        for start in range(0, len(rows), args.batch_size):
                            selected = rows[start:start + args.batch_size]
                            assert all(row['messages'][-1]['role'] == 'assistant' for row in selected)
                            prompts = [llm.tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                add_generation_prompt=True, enable_thinking=False, return_dict=False) for row in selected]
                            with factors(mode):
                                answers = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8))
                            for offset, (row, answer) in enumerate(zip(selected, answers, strict=True)):
                                expected = row['messages'][-1]['content']
                                records.append({'sample_id': start + offset, 'expected': expected,
                                    'correct': answer['text'].strip() == expected.strip(), **answer})
                            result['quality'][mode][name] = {'status': 'MEASURING', 'samples': len(records),
                                'correct': sum(row['correct'] for row in records), 'records': records}
                            save()
                            print(f'quality {mode} {name}: {len(records)}/{len(rows)}', flush=True)
                        result['quality'][mode][name].update(status='MEASURED',
                            accuracy=sum(row['correct'] for row in records) / len(records))
                result['quality_adapter_canary_improvement'] = (
                    result['quality']['enabled']['canary-holdout.jsonl']['correct']
                    > result['quality']['disabled']['canary-holdout.jsonl']['correct'])
                result['quality_replay_exact_matches_preserved'] = (
                    result['quality']['enabled']['replay.jsonl']['correct']
                    >= result['quality']['disabled']['replay.jsonl']['correct'])
                result['native_minimum_canary_accuracy'] = args.minimum_canary_accuracy
                result['native_canary_threshold_met'] = (args.minimum_canary_accuracy is None or
                    result['quality']['enabled']['canary-holdout.jsonl']['accuracy'] >= args.minimum_canary_accuracy)
                save()
                assert result['quality_adapter_canary_improvement'], 'Native adapter did not improve heldout canary answers'
                assert result['quality_replay_exact_matches_preserved'], 'Native adapter reduced exact replay answer matches'
                assert result['native_canary_threshold_met'], 'Native adapter did not meet the supplied canary target'
        if args.check_cache_rebuild:
            original_pages = llm.engine.num_pages
            original_slots = llm.engine.linear_state_pool.num_slots - 1
            identity = llm.engine.model._lora_identity
            result['cache_rebuilds'] = []
            capture_logits = True
            with torch.inference_mode():
                for pages, slots in [(1536, max(16, original_slots + 2)), (original_pages, original_slots)]:
                    llm.rebuild_cache(num_pages=pages, num_mamba_slots=slots)
                    assert llm.engine.model._lora_identity == identity
                    checks = []
                    for index, row in enumerate(outputs):
                        logits.clear()
                        answer = llm.generate([row['prompt_tokens']], SamplingParams(temperature=0, max_tokens=8))[0]
                        saved = torch.load(args.output / f'logits-{index}.pt', weights_only=True)
                        checks.append(answer == row['modes']['enabled']['answer'] and
                                      torch.equal(logits[0].view(torch.uint8), saved['enabled'].view(torch.uint8)))
                    result['cache_rebuilds'].append({'pages': pages, 'usable_GDN_slots': slots,
                        'all_answers_and_logits_E2': all(checks), 'adapter_identity_unchanged': True})
                    save()
                    assert all(checks), 'Cache rebuild changed native answers/logits'
        result['banks_unchanged'] = before_hashes == manager.hashes()
        result['streaming_after_requests'] = manager.inventory()
        result['memory'] = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                            'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
                            'free': torch.cuda.mem_get_info()[0]}
        if args.reference:
            reference = json.loads((args.reference / 'results.json').read_text(encoding='utf-8'))
            checks = []
            for index, (row, previous) in enumerate(zip(outputs, reference['outputs'], strict=True)):
                old_logits = torch.load(args.reference / f'logits-{index}.pt', weights_only=True)
                new_logits = torch.load(args.output / f'logits-{index}.pt', weights_only=True)
                checks.append(row['prompt_tokens'] == previous['prompt_tokens']
                    and all(row['modes'][mode]['answer'] == previous['modes'][mode]['answer']
                            and torch.equal(new_logits[mode].view(torch.uint8), old_logits[mode].view(torch.uint8)) for mode in new_logits))
            result['fresh_process_all_answers_and_logits_E2'] = all(checks)
            assert all(checks)
            if args.corpus:
                quality_equal = all(
                    result['quality'][mode][name]['records'] == reference['quality'][mode][name]['records']
                    for mode in ['disabled', 'enabled'] for name in ['canary-holdout.jsonl', 'replay.jsonl'])
                result['fresh_process_all_quality_answers_E2'] = quality_equal
                assert quality_equal, 'Fresh native engine changed corpus answers'
        assert result['banks_unchanged']
        result['status'] = 'PASS'
        save()
    except Exception:
        result.update(status='FAIL', traceback=traceback.format_exc())
        save()
        raise
    finally:
        if llm is not None:
            llm.shutdown()


if __name__ == '__main__':
    main()
