"""Complete actual BF16 decoder blocks, including all 256 routed experts."""
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import safetensors
import torch

pytestmark=[pytest.mark.needs_weights,pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')]


@pytest.fixture(scope='module',params=[0,3])
def block_pair(request):
    location=os.environ.get('CLOVER_HF_SOURCE')
    if not location: pytest.skip('set CLOVER_HF_SOURCE to the local BF16 revision')
    source=Path(location)
    harness=runpy.run_path(str(Path(__file__).parents[2]/'benchmarks/clover_hf_reference.py'))
    official,configuration,_=harness['official_modules'](source)
    cfg=configuration.CloverMoeConfig(**json.loads((source/'config.json').read_text()))
    from freetoken.distributed import set_tp_info,try_get_tp_info
    from freetoken.layers.quantization import finalize_quant
    from freetoken.layers import rotary
    from freetoken.models.clover_moe.config import parse_config
    from freetoken.models.clover_moe.model import CloverPlusOneRMSNorm
    from freetoken.models.qwen3_5_moe.model import Qwen3_5DecoderLayer
    from freetoken.models.qwen3_5_moe.weight import _DenseReader
    from freetoken.models.register import get_model_spec
    from freetoken.utils.torch_utils import torch_dtype
    if try_get_tp_info() is None: set_tp_info(rank=0,size=1)
    layer=request.param
    prefix=f'model.layers.{layer}'
    config=parse_config(cfg)
    rotary.set_rope_device(torch.device('cuda'))
    rotary.get_rope.cache_clear()
    with torch.device('meta'),torch_dtype(torch.bfloat16):
        reference=official.CloverDecoderLayer(cfg,layer)
        native=Qwen3_5DecoderLayer(config,layer,prefix=prefix)
        native.input_layernorm=CloverPlusOneRMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        native.post_attention_layernorm=CloverPlusOneRMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        if native._is_linear:
            native.linear_attn.conv_right_weight=torch.empty(native.linear_attn.conv_dim,1,3)
            for name in ('bd_softmax_logit_scale','bd_softmax_history_bias','bd_softmax_history_c'):
                setattr(native.linear_attn,name,torch.empty(native.linear_attn.num_v_heads))
        else:
            native.self_attn.q_norm=CloverPlusOneRMSNorm(cfg.head_dim,cfg.rms_norm_eps)
            native.self_attn.k_norm=CloverPlusOneRMSNorm(cfg.head_dim,cfg.rms_norm_eps)
    index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    reader=_DenseReader(None,get_model_spec('CloverMoeModelLM'))
    loaded={}
    banks={}
    for key in sorted(index):
        if not key.startswith(prefix+'.'): continue
        with safetensors.safe_open(source/index[key],framework='pt',device='cpu') as handle:
            tensor=handle.get_tensor(key).cuda()
        relative=key[len(prefix)+1:]
        parent,_,leaf=relative.rpartition('.')
        owner=reference.get_submodule(parent) if parent else reference
        setattr(owner,leaf,torch.nn.Parameter(tensor,requires_grad=False))
        if '.mlp.experts.' in key:
            banks['gate_up' if leaf=='gate_up_proj' else 'down']=tensor
        else:
            values=reader.add(key,tensor)
            if values is None: loaded[key]=tensor
            else: loaded.update(values)
    expected=native.state_dict(prefix=prefix)
    assert set(loaded)==set(expected) and not reader.pending
    native.load_state_dict({key:value.to(expected[key].dtype) for key,value in loaded.items()},prefix=prefix)
    native.mlp.experts._expert_banks=banks
    finalize_quant(native)
    return native,reference,cfg,config,layer


@pytest.mark.parametrize('width',[8,16,32])
@torch.inference_mode()
def test_full_bf16_preview_block_matches_official(block_pair,monkeypatch,width):
    native,official,cfg,config,layer=block_pair
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.clover_moe.preview import make_preview_batch,gdn_forward,attention_preview,moe_preview
    import freetoken.core as core
    device=torch.device('cuda');prefix=55
    pool=LinearStatePool(config.linear_attention_group(),2,torch.bfloat16,device,tp_size=1)
    kv=MHAKVCache(cfg.num_key_value_heads,cfg.num_hidden_layers,cfg.head_dim,256,1,torch.bfloat16,device,layer_ids=[3])
    table=torch.arange(256,device=device,dtype=torch.int32).reshape(1,-1)
    ctx=SimpleNamespace(linear_state_pool=pool,kv_cache=kv,page_table=table)
    engine=SimpleNamespace(device=device,max_seq_len=256,page_table=table)
    req=SimpleNamespace(cached_len=prefix,device_len=prefix+width,linear_slot_idx=1,table_idx=0)
    batch=make_preview_batch(engine,req,[151669]*width)
    ctx.batch=batch
    monkeypatch.setattr(core,'_GLOBAL_CTX',ctx)
    torch.manual_seed(layer+width)
    hidden=torch.randn(width,cfg.hidden_size,device=device,dtype=torch.bfloat16)*.1
    valid=torch.ones(1,width,device=device,dtype=torch.bool)
    if native._is_linear:
        local=pool.local_index(layer)
        pool.conv_states[local,1].normal_(std=.05)
        pool.recurrent_states[local,1].normal_(std=.01)
        cache=dict(conv=pool.conv_states[local,1:2].clone(),ssm=pool.recurrent_states[local,1:2].clone())
    else:
        key=torch.randn(1,cfg.num_key_value_heads,prefix,cfg.head_dim,device=device,dtype=torch.bfloat16)*.1
        value=torch.randn_like(key)
        kv.k_cache(3).view(-1,cfg.num_key_value_heads,cfg.head_dim)[:prefix].copy_(key[0].transpose(0,1))
        kv.v_cache(3).view(-1,cfg.num_key_value_heads,cfg.head_dim)[:prefix].copy_(value[0].transpose(0,1))
        cache=dict(key=key,value=value)
    expected,_=official(hidden.unsqueeze(0),cache,valid,torch.ones(1,prefix,device=device,dtype=torch.bool),
                         batch.positions.reshape(1,-1),True)
    normalized=native.input_layernorm.forward(hidden.clone())
    mixed=gdn_forward(native.linear_attn,normalized,ctx,preview=True) if native._is_linear else attention_preview(native.self_attn,normalized,ctx)
    mixed,residual=native.post_attention_layernorm.forward_add_residual(mixed,hidden.clone())
    actual=moe_preview(native.mlp,mixed,ctx)+residual
    error=float((actual.float()-expected[0].float()).norm()/expected.float().norm())
    print(json.dumps(dict(layer=layer,width=width,relative_l2=error)))
    assert error < .01
