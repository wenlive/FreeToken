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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from freetoken.core import SamplingParams
    from freetoken.layers.base import BaseOP
    from freetoken.llm import LLM
    from freetoken.lora.ops import PackedLinear
    from freetoken.mm.config import MultimodalConfig

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = dict(moe_strategy='fused', dense_layer_streaming=True, max_running_req=1,
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
        logits = []
        original_forward = llm.engine.model.forward
        def forward():
            output = original_forward()
            if not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite real dense inference logits')
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
