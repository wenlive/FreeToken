"""Real-operand attention audit and accumulation-order noise floor; no pass threshold tuning."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.kernel.triton import attention
from freetoken.kernel.triton.clover_attention import split_kv_attention
from freetoken.models.clover_moe import preview


def errors(actual, reference):
    actual, reference = actual.float(), reference.float()
    difference = actual - reference
    return dict(relative_l2=float(difference.norm() / reference.norm().clamp_min(1e-20)),
        max_abs=float(difference.abs().max()), rms=float(difference.square().mean().sqrt()),
        finite=bool(torch.isfinite(actual).all()))


def distribution(actual, reference):
    result = errors(actual, reference)
    p, q = reference.float().log_softmax(-1), actual.float().log_softmax(-1)
    result.update(argmax_agreement=float((actual.argmax(-1) == reference.argmax(-1)).float().mean()),
        kl_per_row=(p.exp() * (p - q)).sum(-1).tolist(),
        argmax_per_row=actual.argmax(-1).tolist())
    return result


@contextmanager
def tile_noise(enabled):
    original = attention._select_extend_tile
    if enabled:
        def perturbed(*args):
            m, n = original(*args)
            if n <= 16:
                raise ValueError('Noise experiment needs a baseline BLOCK_N above 16')
            return m, n // 2
        attention._select_extend_tile = perturbed
    try:
        yield
    finally:
        attention._select_extend_tile = original


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefixes', default='16384,32768')
    p.add_argument('--widths', default='8,16,32')
    p.add_argument('--coresets', default='.9,1.0')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    prefixes, widths = list(map(int, a.prefixes.split(','))), list(map(int, a.widths.split(',')))
    llm = LLM(a.model, moe_strategy='offload', moe_cache_size=2304, expert_load='serial',
        max_running_req=1, memory_ratio=.95, clover_max_width=0,
        num_token_override=max(prefixes) + 128, max_seq_len_override=max(prefixes) + 128,
        max_extend_tokens=4096, cuda_graph_bs=[1])
    engine = llm.engine
    full_layers = [layer.self_attn.layer_id for layer in engine.model.model.layers.op_list if not layer._is_linear]
    selected_layers = {full_layers[0], full_layers[len(full_layers) // 2]}
    result = dict(state='running', operator_cases=[], model_cases=[], selected_layers=sorted(selected_layers),
        protocol='Same prefill state and input; default vs half BLOCK_N vs split8. '
        'FP32 per-head softmax attention on true BF16 operands, TF32 disabled. No acceptance threshold adjusted.')
    current_prefix = None
    capture = False
    current_layer = None
    operands = {}
    original_attention = preview.attention_preview
    original_extend = preview.extend_paged_attention

    def save():
        (a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    def layer_attention(mixer, hidden, ctx, **kw):
        nonlocal current_layer
        current_layer = mixer.layer_id
        return original_attention(mixer, hidden, ctx, **kw)

    def record_extend(q, keys, values, qo, kv, indices, prefix_lens, width, scale, **kw):
        if capture and current_layer in selected_layers:
            prefix = int(prefix_lens[0])
            locations = indices[:prefix].long()
            operands[current_layer] = dict(q=q.cpu(), keys=keys[locations].cpu(), values=values[locations].cpu(),
                k=kw['k_extend'].cpu(), v=kw['v_extend'].cpu(), scale=scale)
        return original_extend(q, keys, values, qo, kv, indices, prefix_lens, width, scale, **kw)

    preview.attention_preview, preview.extend_paged_attention = layer_attention, record_extend
    forward = engine.forward_batch

    def operator_audit(width):
        for layer, stored in operands.items():
            artifact = a.output / f'operands-{current_prefix}-{width}-{layer}.pt'
            torch.save(stored, artifact)
            q, kc, vc, k, v = [stored[key].to(engine.device) for key in ('q', 'keys', 'values', 'k', 'v')]
            prefix = len(kc)
            qo = torch.tensor([0, width], device=engine.device, dtype=torch.int32)
            kv = torch.tensor([0, prefix + width], device=engine.device, dtype=torch.int32)
            indices = torch.arange(prefix, device=engine.device, dtype=torch.int32)
            lens = torch.tensor([prefix], device=engine.device, dtype=torch.int32)
            ends = torch.full((width,), prefix + width, device=engine.device, dtype=torch.int32)
            outputs = {}
            for name, noise in [('default', False), ('tile_noise', True)]:
                with tile_noise(noise):
                    outputs[name] = original_extend(q, kc, vc, qo, kv, indices, lens, width,
                        stored['scale'], k_extend=k, v_extend=v, block_ends=ends).float()
            outputs['split8'] = split_kv_attention(q, kc, vc, indices, lens, k, v, preview=True).float()
            reference = torch.empty_like(q, dtype=torch.float32)
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                for head in range(q.shape[1]):
                    owner = head // (q.shape[1] // k.shape[1])
                    keys = torch.cat((kc[:, owner], k[:, owner])).float()
                    values = torch.cat((vc[:, owner], v[:, owner])).float()
                    reference[:, head] = ((q[:, head].float() @ keys.T) * stored['scale']).softmax(-1) @ values
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
            result['operator_cases'].append(dict(prefix=prefix, width=width, layer=layer,
                artifact=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                shape=list(q.shape), errors_vs_fp32={name: errors(value, reference) for name, value in outputs.items()},
                split_vs_default=errors(outputs['split8'], outputs['default']),
                noise_vs_default=errors(outputs['tile_noise'], outputs['default'])))
            save()

    def persistent_hash(req):
        digest = hashlib.sha256()
        pool = engine.linear_state_pool
        slot = req.linear_slot_idx
        for tensor in (pool.conv_states[:, slot], pool.recurrent_states[:, slot]):
            digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        locations = engine.page_table[req.table_idx, :req.cached_len].long()
        for layer in full_layers:
            for cache in (engine.kv_cache.k_cache(layer), engine.kv_cache.v_cache(layer)):
                tensor = cache.view(-1, *cache.shape[-2:])[locations].contiguous()
                digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    def audited(batch, sampling):
        nonlocal capture
        output = forward(batch, sampling)
        req = batch.reqs[0]
        if not batch.is_prefill or req.cached_len != current_prefix:
            return output
        before = persistent_hash(req)
        for width in widths:
            for anchored in (False, True):
                tokens = torch.full((width,), engine.mtp_runtime.mask_id, device=engine.device, dtype=torch.int32)
                if anchored:
                    tokens[:1].copy_(output.next_tokens_gpu[:1])
                for coreset in map(float, a.coresets.split(',')):
                    logits = {}
                    for name, splits, noise in [('default', 1, False), ('tile_noise', 1, True), ('split8', 8, False)]:
                        temporary = preview.make_preview_batch(engine, req, tokens, coreset_top_p=coreset)
                        temporary.clover_kv_splits = splits
                        capture = name == 'default' and not anchored and coreset == float(a.coresets.split(',')[0])
                        with tile_noise(noise), engine.ctx.forward_batch(temporary):
                            logits[name] = engine.model.forward().float().cpu()
                        capture = False
                    result['model_cases'].append(dict(prefix=current_prefix, width=width, anchored=anchored,
                        coreset=coreset, split_vs_default=distribution(logits['split8'], logits['default']),
                        noise_vs_default=distribution(logits['tile_noise'], logits['default'])))
                    save()
                if not anchored:
                    operator_audit(width)
                    operands.clear()
        after = persistent_hash(req)
        result.setdefault('state_hashes', []).append(dict(prefix=current_prefix, before=before, after=after))
        save()
        assert before == after, 'Attention diagnostics changed persistent state'
        return output

    engine.forward_batch = audited
    match = llm.cache_manager.match_req
    llm.cache_manager.match_req = lambda req, max_len=None: match(req, max_len=0)
    try:
        seed = llm.tokenizer.apply_chat_template([dict(role='user', content='Describe a tree.')],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        pattern = llm.tokenizer.encode('Trees shade the streets. ')
        for current_prefix in prefixes:
            middle = current_prefix - len(seed)
            ids = seed[:-8] + (pattern * ((middle + len(pattern) - 1) // len(pattern)))[:middle] + seed[-8:]
            result.setdefault('inputs', []).append(dict(prefix=current_prefix, token_ids=ids))
            llm.generate([ids], SamplingParams(temperature=0, max_tokens=1, clover_mode='causal'))
        result['state'] = 'complete'
        save()
    finally:
        preview.attention_preview, preview.extend_paged_attention = original_attention, original_extend
        engine.shutdown()


if __name__ == '__main__':
    main()
