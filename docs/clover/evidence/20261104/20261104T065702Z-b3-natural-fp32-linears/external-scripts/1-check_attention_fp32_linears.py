"""Natural-prefix attention control with exact dequantized FP32 preview linears."""
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from freetoken.core import SamplingParams, get_global_ctx
from freetoken.kernel.triton import attention
from freetoken.kernel import moe_sum_reduce_triton
from freetoken.layers.activation import silu_and_mul
from freetoken.models.clover_moe.packed import CloverPackedLinear
from freetoken.moe import fused_gguf as moe
from benchmarks.clover.clover_gguf_decode import decode_expert
from benchmarks.clover.harness import load_engine, save_results
from benchmarks.clover.oracles import make_preview_batch


@contextmanager
def fp32_preview_linears():
    dense_original, moe_original = CloverPackedLinear.forward, moe.fused_experts_gguf
    host, device = {}, OrderedDict()
    device_bytes = 0

    def dense(self, value):
        nonlocal device_bytes
        if not getattr(get_global_ctx().batch, 'clover_preview', False):
            return dense_original(self, value)
        identity = id(self)
        if identity not in host:
            host[identity] = decode_expert(self.weight.cpu().numpy(), self._type, device='cpu')
        if identity not in device:
            weight = host[identity]
            size = weight.numel() * weight.element_size()
            while device and device_bytes + size > 128 << 20:
                _, evicted = device.popitem(last=False)
                device_bytes -= evicted.numel() * evicted.element_size()
            device[identity] = weight.cuda()
            device_bytes += size
        device.move_to_end(identity)
        return F.linear(value.float(), device[identity]).to(value.dtype)

    def experts(hidden, gate_up, down, weights, ids, activation, *, gate_up_type, down_type):
        if not getattr(get_global_ctx().batch, 'clover_preview', False):
            return moe_original(hidden, gate_up, down, weights, ids, activation,
                gate_up_type=gate_up_type, down_type=down_type)
        assert activation == 'silu', activation
        rows, width = hidden.shape
        routed = torch.empty(rows, ids.shape[1], width, device=hidden.device, dtype=hidden.dtype)
        for expert in ids.unique().cpu().tolist():
            row, route = (ids == expert).nonzero(as_tuple=True)
            gu = decode_expert(gate_up[expert].cpu().numpy(), gate_up_type, device=hidden.device)
            intermediate = silu_and_mul(F.linear(hidden[row].float(), gu).to(hidden.dtype))
            dw = decode_expert(down[expert].cpu().numpy(), down_type, device=hidden.device)
            value = F.linear(intermediate.float(), dw)
            routed[row, route] = (value * weights[row, route, None].float()).to(hidden.dtype)
        result = torch.empty_like(hidden)
        moe_sum_reduce_triton(routed, result)
        return result

    CloverPackedLinear.forward, moe.fused_experts_gguf = dense, experts
    try:
        yield
    finally:
        CloverPackedLinear.forward, moe.fused_experts_gguf = dense_original, moe_original


def compare(actual, reference):
    delta = actual - reference
    p, q = reference.log_softmax(-1), actual.log_softmax(-1)
    return dict(relative_l2=float(delta.norm() / reference.norm()), max_abs=float(delta.abs().max()),
        argmax_agreement=float((actual.argmax(-1) == reference.argmax(-1)).float().mean()),
        kl_per_row=(p.exp() * (p - q)).sum(-1).tolist())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--prefixes', default='8192,16384,32768')
    p.add_argument('--width', type=int, default=16)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    prefixes = list(map(int, a.prefixes.split(',')))
    result = dict(state='running', cases=[], inputs=[],
        corpus_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
        protocol='Native clean prefill held fixed; natural GSM8K examples, greedy anchor, coreset=1.0. '
        'Exact dequantized FP32 packed-Q8 dense projections and routed MoE matmul, no Q8_1 activation rounding, TF32 disabled. '
        'Unquantized linears, BF16 activation boundaries and discrete top-8 expert routing remain. '
        'This isolates packed linear arithmetic; it is not an all-FP32 model or a speed backend.')
    llm = load_engine(a.model, context=max(prefixes)+128, cache_slots=2048, memory_ratio=.95,
        max_extend_tokens=4096, clover_max_width=0)
    engine, runtime = llm.engine, llm.engine.mtp_runtime
    match = llm.cache_manager.match_req
    llm.cache_manager.match_req = lambda req, max_len=None: match(req, max_len=0)
    forward = engine.forward_batch
    select = attention._select_extend_tile
    current = None

    def snapshot(req):
        digest = hashlib.sha256()
        pool = engine.linear_state_pool
        for value in (pool.conv_states[:, req.linear_slot_idx], pool.recurrent_states[:, req.linear_slot_idx]):
            digest.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        locations = engine.page_table[req.table_idx, :req.cached_len].long()
        for layer in engine.model.model.layers.op_list:
            if layer._is_linear:
                continue
            for cache in (engine.kv_cache.k_cache(layer._layer_id), engine.kv_cache.v_cache(layer._layer_id)):
                value = cache.view(-1, *cache.shape[-2:])[locations]
                digest.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    def noisy(*args):
        m, n = select(*args)
        assert n > 16
        return m, n // 2

    def observed(batch, sampling):
        output = forward(batch, sampling)
        req = batch.reqs[0]
        if not batch.is_prefill or req.cached_len != current['prefix']:
            return output
        before = snapshot(req)
        tokens = torch.full((a.width,), runtime.mask_id, device='cuda', dtype=torch.int32)
        tokens[:1].copy_(output.next_tokens_gpu[:1])
        scores, artifacts = {}, {}
        with fp32_preview_linears():
            for name, splits, perturb in [('default', 1, False), ('noise', 1, True), ('split8', 8, False)]:
                attention._select_extend_tile = noisy if perturb else select
                preview = make_preview_batch(engine, req, tokens, coreset_top_p=1.0)
                preview.clover_kv_splits = splits
                with engine.ctx.forward_batch(preview):
                    scores[name] = engine.model.forward().float().cpu()
                path = a.output / f"{current['example']}-{current['prefix']}-{name}.pt"
                torch.save(scores[name], path)
                artifacts[name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        attention._select_extend_tile = select
        after = snapshot(req)
        result['cases'].append(dict(**current, width=a.width, block_ids=tokens.cpu().tolist(),
            split_vs_default=compare(scores['split8'], scores['default']),
            noise_vs_default=compare(scores['noise'], scores['default']),
            artifacts=artifacts, state_before=before, state_after=after))
        save_results(a.output, result)
        assert before == after
        return output

    engine.forward_batch = observed
    try:
        rows = [json.loads(line) for line in a.corpus.read_text(encoding='utf-8').splitlines()]
        seed = llm.tokenizer.apply_chat_template([dict(role='user', content='Summarize the methods in these worked examples.')],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        for example, offset in enumerate((0, 400)):
            text = '\n\n'.join(row['question'] + '\n' + row['answer'] for row in rows[offset:offset+400])
            body = llm.tokenizer.encode(text, add_special_tokens=False)
            for prefix in prefixes:
                ids = seed[:-8] + body[:prefix-len(seed)] + seed[-8:]
                assert len(ids) == prefix
                current = dict(example=example, prefix=prefix)
                result['inputs'].append(dict(**current, token_ids=ids))
                llm.generate([ids], SamplingParams(temperature=0, max_tokens=1, clover_mode='causal'))
        result['state'] = 'complete'
        save_results(a.output, result)
    finally:
        attention._select_extend_tile = select
        engine.shutdown()


if __name__ == '__main__':
    main()
