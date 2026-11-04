"""Locate native/HF prefill drift and compare chunk GDN on identical real operands."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from benchmarks.clover.harness import PROMPTS, load_engine, save_results
from benchmarks.clover.clover_hf_reference import official_modules


def errors(actual, reference):
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    return dict(relative_l2=float(delta.norm()/reference.norm().clamp_min(1e-20)),
        max_abs=float(delta.abs().max()), finite=bool(torch.isfinite(actual).all()))


def compare_layers(actual, reference):
    rows = []
    for index, (left, right) in enumerate(zip(actual, reference, strict=True)):
        assert left.keys() == right.keys(), (index, left.keys(), right.keys())
        metrics = {name: errors(left[name], right[name]) for name in left}
        aid = left['router'].float().argsort(-1, descending=True, stable=True)[..., :8]
        bid = right['router'].float().argsort(-1, descending=True, stable=True)[..., :8]
        rows.append(dict(layer=index, errors=metrics,
            top8_set_agreement=float((aid.sort(-1).values == bid.sort(-1).values).all(-1).float().mean()),
            top8_member_fraction=float((aid[..., :, None] == bid[..., None, :]).any(-1).float().mean())))
    first = {}
    for limit in (.001, .01):
        first[str(limit)] = {name: next((row['layer'] for row in rows
            if row['errors'].get(name, {}).get('relative_l2', 0) > limit), None)
            for name in ('hidden', 'mixer_input', 'mixer_output', 'mlp_output', 'conv', 'ssm', 'key', 'value')}
    return dict(layers=rows, first_layer_exceeding_diagnostic_levels=first)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['native', 'reference'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-results', type=Path, nargs='+')
    p.add_argument('--noise', choices=['none', 'tile'], default='none')
    p.add_argument('--sparse-limit', type=int, default=128)
    p.add_argument('--prompts', default='Count,Sky,Python,JSON,Chinese')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(state='running', stage=a.stage, cases=[], gdn_operator_cases=[],
        protocol='Diagnostic only: no acceptance threshold changes. Per-layer native vs same-Q8 HF and native tile/sparse controls. '
        '1e-3 and 1e-2 are reporting landmarks, not pass/fail tolerances.')
    official = official_modules(a.source)[0]

    def save():
        save_results(a.output, result)

    def artifact(name, payload):
        path = a.output / name
        torch.save(payload, path)
        return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    if a.stage == 'reference':
        from benchmarks.clover.clover_gguf_reference import load_gguf_official
        assert a.native_results
        native = [(path, json.loads(path.read_bytes())) for path in a.native_results]
        model, storage = load_gguf_official(a.source, a.model, cache_bytes=4 << 30)
        trace = []
        handles = []
        for index, layer in enumerate(model.model.layers):
            def capture_layer(module, inputs, output, index=index):
                value, cache = output
                trace[index].update(hidden=value.cpu(), **{key: v.cpu() for key,v in cache.items()})
            handles.append(layer.register_forward_hook(capture_layer))
            mixer = layer.linear_attn if layer.preview_type == 'linear_attention' else layer.self_attn
            def capture_mixer(module, inputs, output, index=index):
                trace[index].update(mixer_input=inputs[0].cpu(), mixer_output=output[0].cpu())
            handles.append(mixer.register_forward_hook(capture_mixer))
            def capture_mlp(module, inputs, output, index=index):
                trace[index].update(mlp_input=inputs[0].cpu(), mlp_output=output.cpu())
            handles.append(layer.mlp.register_forward_hook(capture_mlp))
            def capture_router(module, inputs, output, index=index):
                trace[index]['router'] = output.cpu()
            handles.append(layer.mlp.gate.register_forward_hook(capture_router))
        try:
            for reference_case in native[0][1]['cases']:
                trace = [{} for _ in model.model.layers]
                ids = reference_case['prompt_ids']
                output = model.model(torch.tensor([ids], device='cuda'), use_cache=True)
                cache_artifact = artifact(reference_case['prompt'] + '-hf.pt', trace)
                comparisons, natives = [], []
                for path, data in native:
                    case = next(row for row in data['cases'] if row['prompt'] == reference_case['prompt'])
                    assert case['prompt_ids'] == ids
                    native_file = path.parent / case['artifact']['file']
                    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == case['artifact']['sha256']
                    actual = torch.load(native_file, weights_only=True)
                    natives.append(actual)
                    comparisons.append(dict(native=str(path), artifact_sha256=case['artifact']['sha256'],
                        comparison=compare_layers(actual, trace)))
                noise = [dict(native=str(native[index][0]), comparison=compare_layers(actual, natives[0]))
                    for index, actual in enumerate(natives[1:], 1)]
                result['cases'].append(dict(prompt=reference_case['prompt'], artifact=cache_artifact,
                    against_hf=comparisons, native_controls=noise))
                save()
                del output
        finally:
            for handle in handles:
                handle.remove()
        result['state'] = 'complete'
        save()
        return

    from freetoken.core import SamplingParams, get_global_ctx
    from freetoken.kernel.triton import attention
    from freetoken.models.qwen3_5_moe import gdn
    llm = load_engine(a.model, context=4096, cache_slots=2292, memory_ratio=.9,
        clover_max_width=0, clover_sparse_prefill_limit=a.sparse_limit)
    engine = llm.engine
    match = llm.cache_manager.match_req
    llm.cache_manager.match_req = lambda req, max_len=None: match(req, max_len=0)
    current, layer_index = None, None
    trace = []
    select, chunk = attention._select_extend_tile, gdn.gdn_prefill_chunk_fla
    originals = []

    def noisy(*args):
        m, n = select(*args)
        if get_global_ctx().batch.is_prefill:
            assert n > 16
            return m, n // 2
        return m, n

    def record_chunk(q, k, v, g, beta, **options):
        active = current is not None and get_global_ctx().batch.is_prefill
        if active:
            initial = options['state_source'][options['indices'].long()].clone()
        output = chunk(q, k, v, g, beta, **options)
        if active:
            actual = output[0] if isinstance(output, tuple) else output
            actual_state = options['state_source'][options['indices'].long()].clone()
            valid = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
            expected, expected_state = official.recurrent_commit_torch(q, k, v, g, beta, initial, valid)
            row = dict(prompt=current, layer=layer_index, tokens=q.shape[1],
                output_vs_hf=errors(actual.unsqueeze(0), expected), state_vs_hf=errors(actual_state, expected_state))
            if layer_index in (0, 20, 38):
                row['artifact'] = artifact(f'{current}-gdn-{layer_index}.pt', dict(q=q.cpu(), k=k.cpu(),
                    v=v.cpu(), g=g.cpu(), beta=beta.cpu(), initial=initial.cpu(),
                    actual=actual.cpu(), actual_state=actual_state.cpu(), expected=expected.cpu(), expected_state=expected_state.cpu()))
            result['gdn_operator_cases'].append(row)
        return output

    def wrap(owner, name, callback):
        old = getattr(owner, name)
        originals.append((owner, name, old))
        setattr(owner, name, callback(old))

    for index, layer in enumerate(engine.model.model.layers.op_list):
        def layer_wrapper(old, index=index, layer=layer):
            def observed(hidden, residual):
                nonlocal layer_index
                layer_index = index
                value, resid = old(hidden, residual)
                batch = get_global_ctx().batch
                if current is not None and batch.is_prefill:
                    req = batch.reqs[0]
                    trace[index]['hidden'] = (value + resid).unsqueeze(0).cpu()
                    if layer._is_linear:
                        pool = engine.linear_state_pool
                        local, slot = pool.local_index(index), req.linear_slot_idx
                        trace[index].update(conv=pool.conv_states[local, slot:slot+1].cpu(),
                            ssm=pool.recurrent_states[local, slot:slot+1].cpu())
                    else:
                        mixer = layer.self_attn
                        locations = engine.page_table[req.table_idx, :req.device_len].long()
                        for key, cache in [('key', engine.kv_cache.k_cache(index)), ('value', engine.kv_cache.v_cache(index))]:
                            trace[index][key] = cache.view(-1, mixer.num_kv, mixer.head_dim)[locations].transpose(0, 1).unsqueeze(0).cpu()
                return value, resid
            return observed
        wrap(layer, 'forward', layer_wrapper)
        for label, module in [('mixer', layer.linear_attn if layer._is_linear else layer.self_attn), ('mlp', layer.mlp)]:
            def module_wrapper(old, index=index, label=label):
                def observed(value):
                    active = current is not None and get_global_ctx().batch.is_prefill
                    if active:
                        input_value = value.unsqueeze(0).cpu()
                    output = old(value)
                    if active:
                        trace[index][label + '_input'] = input_value
                        trace[index][label + '_output'] = output.unsqueeze(0).cpu()
                    return output
                return observed
            wrap(module, 'forward', module_wrapper)
        def router_wrapper(old, index=index):
            def observed(value):
                output = old(value)
                if current is not None and get_global_ctx().batch.is_prefill:
                    trace[index]['router'] = output.unsqueeze(0).cpu()
                return output
            return observed
        wrap(layer.mlp.gate, 'forward', router_wrapper)
    gdn.gdn_prefill_chunk_fla = record_chunk
    if a.noise == 'tile':
        attention._select_extend_tile = noisy
    try:
        for current in a.prompts.split(','):
            trace = [{} for _ in engine.model.model.layers.op_list]
            ids = llm.tokenizer.apply_chat_template([dict(role='user', content=PROMPTS[current])],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            llm.generate([ids], SamplingParams(temperature=0, max_tokens=1, clover_mode='causal'))
            assert len(trace) == 40 and all('hidden' in row and 'router' in row for row in trace)
            result['cases'].append(dict(prompt=current, prompt_ids=ids, noise=a.noise, sparse_limit=a.sparse_limit,
                artifact=artifact(current + '-native.pt', trace)))
            save()
        result['state'] = 'complete'
        save()
    finally:
        attention._select_extend_tile, gdn.gdn_prefill_chunk_fla = select, chunk
        for owner, name, old in originals:
            setattr(owner, name, old)
        engine.shutdown()


if __name__ == '__main__':
    main()
