"""Paired preview arithmetic diagnostics; fixed weights, inputs and clean prefix.

No reference arithmetic in this file is a serving backend or a timing benchmark.
Dense/MoE oracle weights remain exactly dequantized FP32, avoiding BF16 weight rounding.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from clover_gguf_decode import decode_expert
from clover_gguf_reference import q8_1_activations, load_gguf_official
from clover_hf_reference import official_modules


def metrics(actual, expected, mask):
    a, b = actual.float().clone(), expected.float().clone()
    a[:, mask] = b[:, mask] = -float('inf')
    p, q = b.log_softmax(-1), a.log_softmax(-1)
    probabilities = p.exp()
    return dict(argmax_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean()),
        mean_kl=float(torch.where(probabilities>0, probabilities*(p-q), 0).sum(-1).mean()),
        threshold_crossings=((a.softmax(-1).amax(-1)>.95)!=(b.softmax(-1).amax(-1)>.95)).nonzero().flatten().tolist())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['native', 'reference'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--native-output', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4)
    result = dict(state='running', scope=__doc__, stage=a.stage, cases=[])
    def save():
        (a.output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    if a.stage == 'reference':
        import clover_gguf_reference as oracle
        native = json.loads((a.native_output/'results.json').read_text(encoding='utf-8'))
        model, storage = load_gguf_official(a.source, a.model, cache_bytes=4<<30)
        for case in native['cases']:
            ids = torch.tensor([case['prompt_ids']], device='cuda')
            block = torch.tensor([case['block_ids']], device='cuda')
            cache = model.model(ids, use_cache=True).past_key_values
            references = {}
            for label, quantize in [('q8_1', q8_1_activations), ('no_activation_quantization', lambda x:x.float())]:
                oracle.q8_1_activations = quantize
                references[label] = model(block, past_key_values=cache, use_cache=True, preview=True).logits[0].float().cpu()
                torch.save(references[label], a.output/f"{case['prompt']}-{label}.pt")
            if 'cache_file' in case:
                native_cache = torch.load(a.native_output/case['cache_file'], weights_only=True)
                shared = type(cache)(layers=[{k:v.cuda() for k,v in layer.items()} for layer in native_cache],
                                     attention_mask=cache.attention_mask)
                for label, quantize in [('q8_1_native_state', q8_1_activations),
                                        ('no_activation_native_state', lambda x:x.float())]:
                    oracle.q8_1_activations = quantize
                    references[label] = model(block, past_key_values=shared, use_cache=True, preview=True).logits[0].float().cpu()
            oracle.q8_1_activations = q8_1_activations
            comparisons = {}
            for label, filename in case['logits'].items():
                actual = torch.load(a.native_output/filename, weights_only=True)
                comparisons[label] = {reference: metrics(actual, expected, case['mask_id'])
                    for reference, expected in references.items()}
            row = dict(prompt=case['prompt'], width=case['width'], comparisons=comparisons)
            result['cases'].append(row)
            print(json.dumps(row), flush=True)
            save()
        result['state'] = 'complete'
        save()
        return

    from freetoken.core import SamplingParams, get_global_ctx
    from freetoken.llm import LLM
    from freetoken.models.clover_moe.preview import make_preview_batch
    from freetoken.models.clover_moe.packed import CloverPackedLinear
    from freetoken.layers.activation import silu_and_mul
    from freetoken.kernel import moe_sum_reduce_triton
    import freetoken.models.clover_moe.preview as preview
    import freetoken.moe.fused_gguf as moe
    from bench_clover import PROMPTS

    llm = LLM(a.model, moe_strategy='offload', moe_cache_size=2048, expert_load='serial',
        max_running_req=1, memory_ratio=.95, num_token_override=4096,
        max_seq_len_override=4096, max_extend_tokens=4096, cuda_graph_bs=[1])
    official = official_modules(a.source)[0]
    def torch_conv(value, **options):
        assert options.pop('bias') is None
        assert options.pop('activation') == 'silu'
        assert options.pop('conv_mode') == 'independent_symmetric'
        return official.bd_gdn_conv1d(value, **options)
    dense_original, moe_original = CloverPackedLinear.forward, moe.fused_experts_gguf
    conv_original, gdn_original, coreset_original = (preview.clover_local_memory_conv,
        preview.clover_local_memory_triton, preview.clover_block_coreset_triton)
    active = dict(dense=False, moe=False, quantize=True)
    host_dense, gpu_dense = {}, OrderedDict()
    gpu_bytes = 0

    def dense_forward(self, x):
        nonlocal gpu_bytes
        if not active['dense'] or not getattr(get_global_ctx().batch, 'clover_preview', False):
            return dense_original(self, x)
        identity = id(self)
        if identity not in host_dense:
            host_dense[identity] = decode_expert(self.weight.cpu().numpy(), self._type, device='cpu')
        if identity not in gpu_dense:
            value = host_dense[identity]
            size = value.numel()*value.element_size()
            while gpu_dense and gpu_bytes+size > 128<<20:
                _, evicted = gpu_dense.popitem(last=False)
                gpu_bytes -= evicted.numel()*evicted.element_size()
            gpu_dense[identity] = value.cuda()
            gpu_bytes += size
        gpu_dense.move_to_end(identity)
        value = q8_1_activations(x) if active['quantize'] else x.float()
        return F.linear(value, gpu_dense[identity]).to(x.dtype)

    def expert_forward(hidden, gate_up, down, weights, ids, activation, *, gate_up_type, down_type):
        if not active['moe'] or not getattr(get_global_ctx().batch, 'clover_preview', False):
            return moe_original(hidden, gate_up, down, weights, ids, activation,
                gate_up_type=gate_up_type, down_type=down_type)
        count, width = hidden.shape
        routed = torch.empty(count, ids.shape[1], width, device=hidden.device, dtype=hidden.dtype)
        for expert in ids.unique().cpu().tolist():
            row, route = (ids==expert).nonzero(as_tuple=True)
            gu = decode_expert(gate_up[expert].cpu().numpy(), gate_up_type, device=hidden.device)
            value = hidden[row]
            value = q8_1_activations(value) if active['quantize'] else value.float()
            intermediate = silu_and_mul(F.linear(value, gu).to(hidden.dtype))
            dw = decode_expert(down[expert].cpu().numpy(), down_type, device=hidden.device)
            value = q8_1_activations(intermediate.float()) if active['quantize'] else intermediate.float()
            output = F.linear(value, dw)
            routed[row, route] = (output*weights[row,route,None].float()).to(hidden.dtype)
        output = torch.empty_like(hidden)
        moe_sum_reduce_triton(routed, output)
        return output

    CloverPackedLinear.forward, moe.fused_experts_gguf = dense_forward, expert_forward
    original = llm.engine.forward_batch
    current = None
    def audited(batch, sampling):
        output = original(batch, sampling)
        if batch.is_prefill:
            req = batch.reqs[0]
            name, width, ids = current
            tokens = torch.full((width,), llm.engine.mtp_runtime.mask_id, device='cuda', dtype=torch.int32)
            case = dict(prompt=name, width=width, prompt_ids=ids, block_ids=tokens.cpu().tolist(),
                        mask_id=llm.engine.mtp_runtime.mask_id, logits={})
            prefix_cache = []
            pool = llm.engine.linear_state_pool
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            for layer in llm.engine.model.model.layers.op_list:
                if layer._is_linear:
                    local = pool.local_index(layer._layer_id)
                    prefix_cache.append(dict(conv=pool.conv_states[local,slot:slot+1].cpu(),
                                             ssm=pool.recurrent_states[local,slot:slot+1].cpu()))
                else:
                    mixer = layer.self_attn
                    locations = llm.engine.page_table[req.table_idx,:req.cached_len].long()
                    prefix_cache.append({role:getattr(llm.engine.kv_cache,f'{role[0]}_cache')(layer._layer_id)
                        .view(-1,mixer.num_kv,mixer.head_dim)[locations].transpose(0,1).unsqueeze(0).cpu()
                        for role in ('key','value')})
            case['cache_file'] = f'{name}-prefix-cache.pt'
            torch.save(prefix_cache, a.output/case['cache_file'])
            variants = [('native',False,False,True,False),
                        ('torch_conv',False,False,True,'conv'),
                        ('torch_gdn',False,False,True,'gdn'),
                        ('torch_coreset',False,False,True,'coreset'),
                        ('torch_preview_ops',False,False,True,True),
                        ('f32_q8_1_dense',True,False,True,False),
                        ('f32_q8_1_dense_moe',True,True,True,False),
                        ('all_torch_q8_1',True,True,True,True),
                        ('all_torch_no_activation',True,True,False,True)]
            for label, dense, experts, quantize, torch_ops in variants:
                active.update(dense=dense, moe=experts, quantize=quantize)
                preview.clover_local_memory_conv = torch_conv if torch_ops in (True,'conv') else conv_original
                preview.clover_local_memory_triton = official.bd_gdn_local_memory if torch_ops in (True,'gdn') else gdn_original
                preview.clover_block_coreset_triton = (lambda logits,w,k,p: official.coreset_torch(
                    logits.reshape(1,w,-1), torch.ones(1,w,device=logits.device,dtype=torch.bool), k,p).reshape_as(logits)) if torch_ops in (True,'coreset') else coreset_original
                temporary = make_preview_batch(llm.engine, req, tokens)
                with llm.engine.ctx.forward_batch(temporary):
                    logits = llm.engine.model.forward().float().cpu()
                filename = f'{name}-{label}.pt'
                torch.save(logits, a.output/filename)
                case['logits'][label] = filename
                print(name, label, flush=True)
            active.update(dense=False, moe=False, quantize=True)
            preview.clover_local_memory_conv, preview.clover_local_memory_triton = conv_original, gdn_original
            preview.clover_block_coreset_triton = coreset_original
            result['cases'].append(case)
            save()
        return output
    llm.engine.forward_batch = audited
    try:
        for name, width in [('JSON',16), ('Count',32)]:
            ids = llm.tokenizer.apply_chat_template([dict(role='user',content=PROMPTS[name])],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            current = name,width,ids
            llm.generate([ids], SamplingParams(temperature=0,max_tokens=1,clover_mode='causal'))
        result['state'] = 'complete'
        save()
    finally:
        llm.engine.shutdown()


if __name__ == '__main__':
    main()
