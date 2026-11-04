"""Fixed-history Q8 native/noise/HF/llama distribution diagnostics, never timing evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def read_result(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    assert data['state'] == 'complete', path
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['native', 'noise', 'hf', 'llama', 'compare'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--noise-results', type=Path)
    parser.add_argument('--hf-results', type=Path)
    parser.add_argument('--llama-results', type=Path)
    parser.add_argument('--llama-build', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert args.steps >= 2
    baseline = read_result(args.baseline) if args.baseline else None
    if args.stage != 'native':
        assert baseline is not None
    result = dict(state='running', stage=args.stage, cases=[], model=args.model,
        hf_source=str(args.source), steps=baseline['steps'] if baseline else args.steps,
        protocol='Official Q8, identical forced causal histories and preview inputs. '
        'HF is the semantic reference; llama is an independent numerical ruler. '
        'Native tile perturbation is a measured noise floor. Mask logit excluded consistently. '
        'The 1.2 envelope is reported, not used to silently relax production tolerances.')
    if baseline:
        result['baseline'] = dict(path=str(args.baseline), sha256=file_hash(args.baseline))

    def save():
        (args.output / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    def artifact(name, payload):
        import torch
        path = args.output / name
        torch.save(payload, path)
        return dict(file=path.name, sha256=file_hash(path))

    def load_artifact(path, case):
        import torch
        source = path.parent / case['artifact']['file']
        assert file_hash(source) == case['artifact']['sha256'], source
        return torch.load(source, weights_only=True, map_location='cpu')

    if args.stage == 'llama':
        assert args.llama_build
        manifest_path = args.llama_build / 'build.json'
        build = json.loads(manifest_path.read_text())
        assert build['state'] == 'passed'
        exe = args.llama_build / 'clover-llama-preview.exe'
        assert file_hash(exe).lower() == build['binary_sha256'].lower()
        assert file_hash(args.baseline.parent / 'cases.txt') == baseline['cases_text_sha256']
        stock = Path('C:/workspace/builds/T-20261031-clover-gguf/llama-official-cb09-sm120/bin')
        command = [str(exe), '-m', args.model, '-c', '4096', '-b', '256', '-ub', '256',
            '-ngl', '99', '--cpu-moe', '--moe-cache-mib', '6656', '--load-mode', 'mmap',
            '-t', '8', '-tb', '8', '-ctk', 'bf16', '-ctv', 'bf16', '--flash-attn', 'on',
            '--temp', '0', '--top-k', '1', '--seed', '42', '--repeat-penalty', '1',
            '--presence-penalty', '0', '--frequency-penalty', '0', '--spec-type', 'draft-self',
            '--spec-draft-n-max', '31', '--no-warmup', '--clover-cases',
            str(args.baseline.parent / 'cases.txt'), '--clover-output', str(args.output)]
        result.update(command=command, executable_sha256=file_hash(exe),
            build_manifest=str(manifest_path), build_manifest_sha256=file_hash(manifest_path))
        save()
        with (args.output / 'llama-stdout.log').open('wb') as out, (args.output / 'llama-stderr.log').open('wb') as err:
            child = subprocess.Popen(command, stdout=out, stderr=err,
                env=dict(os.environ, PATH=str(stock) + os.pathsep + os.environ['PATH']))
            result['pid'] = child.pid
            save()
            result['exit_code'] = child.wait()
        save()
        assert result['exit_code'] == 0, result['exit_code']
        reports = [json.loads(line) for line in (args.output / 'llama-stdout.log').read_text().splitlines()
            if line.startswith('{"type":"clover_diagnostic"')]
        assert len(reports) == len(baseline['cases'])
        for index, (report, case) in enumerate(zip(reports, baseline['cases'], strict=True)):
            assert report['case'] == index and report['restored_causal_equal']
            assert report['steps'] == result['steps'] and report['blocks'] == len(case['blocks'])
            files = [dict(file=f'case-{index}-causal.f32', rows=result['steps'])] + [
                dict(file=f'case-{index}-preview-{block}.f32', rows=len(spec['ids']))
                for block, spec in enumerate(case['blocks'])]
            for item in files:
                path = args.output / item['file']
                assert path.stat().st_size == 4 * report['vocab'] * item['rows']
                item['sha256'] = file_hash(path)
            result['cases'].append(dict(**case, llama=report, files=files))
        result['state'] = 'complete'
        save()
        return

    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage == 'compare':
        import numpy as np
        paths = dict(noise=args.noise_results, hf=args.hf_results, llama=args.llama_results)
        others = {name: read_result(path) for name, path in paths.items()}
        for name, data in others.items():
            assert data['baseline']['sha256'] == file_hash(args.baseline), name
            assert len(data['cases']) == len(baseline['cases'])

        def metrics(actual, expected, mask):
            keep = torch.arange(actual.shape[-1]) != mask
            actual, expected = actual[:, keep].float(), expected[:, keep].float()
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            p, q = expected.log_softmax(-1), actual.log_softmax(-1)
            return dict(relative_l2_per_row=((actual-expected).norm(dim=-1) / expected.norm(dim=-1)).tolist(),
                kl_per_row=(p.exp() * (p-q)).sum(-1).clamp_min(0).tolist(),
                top1_equal_per_row=(actual.argmax(-1) == expected.argmax(-1)).tolist())

        for index, case in enumerate(baseline['cases']):
            native = load_artifact(args.baseline, case)
            payload = {name: load_artifact(paths[name], others[name]['cases'][index]) for name in ('noise', 'hf')}
            llama = others['llama']['cases'][index]
            for data in others.values():
                other = data['cases'][index]
                for key in ('prompt_ids', 'forced_ids', 'blocks', 'mask_id'):
                    assert case[key] == other[key], key
            llama_payload = []
            for item in llama['files']:
                path = paths['llama'].parent / item['file']
                assert file_hash(path) == item['sha256']
                llama_payload.append(torch.from_numpy(np.fromfile(path, dtype='<f4').reshape(item['rows'], -1)))
            rows = []
            for part, label in enumerate(['causal'] + [b['name'] for b in case['blocks']]):
                x = native['causal'] if part == 0 else native['previews'][part-1]
                n = payload['noise']['causal'] if part == 0 else payload['noise']['previews'][part-1]
                h = payload['hf']['causal'] if part == 0 else payload['hf']['previews'][part-1]
                l = llama_payload[part]
                assert x.shape == n.shape == h.shape == l.shape
                measures = dict(native_hf=metrics(x,h,case['mask_id']), llama_hf=metrics(l,h,case['mask_id']),
                    native_noise=metrics(n,x,case['mask_id']), native_llama=metrics(x,l,case['mask_id']))
                envelope = {}
                for key in ('relative_l2_per_row', 'kl_per_row'):
                    measured, reference, floor = [measures[name][key] for name in ('native_hf', 'llama_hf', 'native_noise')]
                    limits = [1.2 * max(a,b) for a,b in zip(reference, floor, strict=True)]
                    envelope[key] = dict(bounds=limits, within=[a <= b for a,b in zip(measured, limits, strict=True)])
                rows.append(dict(part=label, metrics=measures, heuristic_envelope=envelope))
            result['cases'].append(dict(prompt=case['prompt'], comparisons=rows))
            save()
        result['state'] = 'complete'
        save()
        return

    if args.stage == 'hf':
        from benchmarks.clover.clover_gguf_reference import load_gguf_official
        model, storage = load_gguf_official(args.source, args.model, cache_bytes=4 << 30)
        for index, case in enumerate(baseline['cases']):
            output = model(torch.tensor([case['prompt_ids']], device='cuda'), use_cache=True)
            clean = output.past_key_values
            clean_copies = [{key: value.clone() for key,value in layer.items()} for layer in clean.layers]
            logits = [output.logits[0,-1].float().cpu()]
            for token in case['forced_ids'][:-1]:
                output = model(torch.tensor([[token]], device='cuda'), past_key_values=output.past_key_values, use_cache=True)
                logits.append(output.logits[0,-1].float().cpu())
            previews = [model(torch.tensor([block['ids']], device='cuda'), past_key_values=clean,
                use_cache=True, preview=True).logits[0].float().cpu() for block in case['blocks']]
            assert all(torch.equal(value, copy[key]) for layer,copy in zip(clean.layers,clean_copies,strict=True)
                for key,value in layer.items()), 'HF clean cache mutated'
            result['cases'].append(dict(**{k:v for k,v in case.items() if k != 'artifact'},
                artifact=artifact(f'case-{index}-hf.pt', dict(causal=torch.stack(logits), previews=previews))))
            save()
        result['state'] = 'complete'
        save()
        return

    from benchmarks.clover.harness import PROMPTS, load_engine
    from benchmarks.clover.oracles import make_preview_batch
    from freetoken.core import SamplingParams
    from freetoken.kernel.triton import attention
    llm = load_engine(args.model, context=4096, cache_slots=2292, memory_ratio=.9,
        clover_max_width=0, gguf_embedding_host=False, clover_disable_penalties=True)
    engine, runtime = llm.engine, llm.engine.mtp_runtime
    original_forward, original_logits, original_sample = engine.forward_batch, runtime.process_logits, engine.sampler.sample
    original_tile = attention._select_extend_tile
    current = None

    def state_hash(req):
        digest = hashlib.sha256()
        pool = engine.linear_state_pool
        for tensor in (pool.conv_states[:,req.linear_slot_idx], pool.recurrent_states[:,req.linear_slot_idx]):
            digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        loc = engine.page_table[req.table_idx,:req.cached_len].long()
        for layer in engine.model.model.layers.op_list:
            if not layer._is_linear:
                for tensor in (engine.kv_cache.k_cache(layer._layer_id), engine.kv_cache.v_cache(layer._layer_id)):
                    value = tensor.view(-1,*tensor.shape[-2:])[loc].contiguous()
                    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    def collect_logits(batch, logits):
        current['causal'].append(logits[0].float().cpu())
        return original_logits(batch, logits)

    def force_sample(logits, sampling):
        if args.stage == 'noise':
            step = min(len(current['causal'])-1, len(current['case']['forced_ids'])-1)
            tokens = torch.tensor([current['case']['forced_ids'][step]], device=logits.device)
        else:
            tokens = original_sample(logits, sampling)
        current['sampled'].append(int(tokens[0]))
        return tokens

    def capture_preview(batch, sampling):
        output = original_forward(batch, sampling)
        req = batch.reqs[0]
        if not batch.is_prefill or req.cached_len != len(current['case']['prompt_ids']):
            return output
        before = state_hash(req)
        def evaluate(spec):
            tokens = torch.tensor(spec['ids'], device=engine.device, dtype=torch.int32)
            temporary = make_preview_batch(engine, req, tokens, coreset_top_p=.9)
            with engine.ctx.forward_batch(temporary):
                scores = engine.model.forward().float().cpu()
            current['previews'].append(scores)
            return scores
        if args.stage == 'noise':
            for spec in current['case']['blocks']:
                evaluate(spec)
        else:
            for width in (8,16,32):
                ids = [runtime.mask_id] * width
                spec = dict(name=f'w{width}-mask', ids=ids.copy())
                current['case']['blocks'].append(spec)
                scores = evaluate(spec).clone()
                scores[:,runtime.mask_id] = -float('inf')
                picks = scores.argmax(-1).tolist()
                ids[0] = int(output.next_tokens_gpu[0])
                spec = dict(name=f'w{width}-anchor', ids=ids.copy())
                current['case']['blocks'].append(spec)
                evaluate(spec)
                for position in range(4,width,4):
                    ids[position] = picks[position]
                spec = dict(name=f'w{width}-partial', ids=ids.copy())
                current['case']['blocks'].append(spec)
                evaluate(spec)
        after = state_hash(req)
        assert before == after, 'Preview mutated persistent prefix state'
        current['case']['prefix_hash'] = dict(before=before, after=after)
        return output

    def noise_tile(*values):
        m,n = original_tile(*values)
        assert n > 16
        return m,n//2

    if args.stage == 'noise':
        attention._select_extend_tile = noise_tile
    engine.forward_batch, runtime.process_logits, engine.sampler.sample = capture_preview, collect_logits, force_sample
    match = llm.cache_manager.match_req
    llm.cache_manager.match_req = lambda req, max_len=None: match(req,max_len=0)
    try:
        source_cases = baseline['cases'] if baseline else [dict(prompt=name) for name in PROMPTS]
        for index, source_case in enumerate(source_cases):
            name = source_case['prompt']
            case = {k:v for k,v in source_case.items() if k != 'artifact'}
            if not baseline:
                case.update(prompt_ids=llm.tokenizer.apply_chat_template([dict(role='user',content=PROMPTS[name])],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False),
                    blocks=[], mask_id=runtime.mask_id)
            current = dict(case=case, causal=[], previews=[], sampled=[])
            output = llm.generate([case['prompt_ids']], SamplingParams(temperature=0,
                max_tokens=result['steps'], ignore_eos=True, clover_mode='causal'))[0]
            ids = current['sampled'][:result['steps']]
            assert len(ids) == result['steps'], (len(ids),result['steps'])
            if baseline:
                assert ids == case['forced_ids'], 'Forced history changed'
            case['forced_ids'] = ids
            case['visible_ids'] = output['token_ids']
            assert result['steps'] <= len(current['causal']) <= result['steps']+1
            case['observed_causal_forwards'] = len(current['causal'])
            case['artifact'] = artifact(f'case-{index}-{args.stage}.pt', dict(
                causal=torch.stack(current['causal'][:result['steps']]), previews=current['previews']))
            result['cases'].append(case)
            save()
        if args.stage == 'native':
            lines = [str(len(result['cases']))]
            for case in result['cases']:
                lines += [f"{len(case['prompt_ids'])} {result['steps']} {len(case['blocks'])}",
                    ' '.join(map(str,case['prompt_ids'])), ' '.join(map(str,case['forced_ids']))]
                for block in case['blocks']:
                    lines += [str(len(block['ids'])), ' '.join(map(str,block['ids']))]
            (args.output / 'cases.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            result['cases_text_sha256'] = file_hash(args.output / 'cases.txt')
        result['state'] = 'complete'
        save()
    finally:
        attention._select_extend_tile = original_tile
        engine.forward_batch, runtime.process_logits, engine.sampler.sample = original_forward, original_logits, original_sample
        engine.shutdown()


if __name__ == '__main__':
    import torch
    with torch.inference_mode():
        main()
