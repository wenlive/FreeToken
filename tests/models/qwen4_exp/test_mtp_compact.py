"""Compare compact transactions against native per-token GDN, prefixes, anchors and graph replay."""
import json
from dataclasses import replace
from types import SimpleNamespace
import torch
from freetoken.core import Context, set_global_ctx
from freetoken.attention.linear import FLAMetadata
from freetoken.kvcache.linear_state_pool import LinearStatePool, state_pool_bytes, _linear_pool_num_slots
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
import pytest


def test_compact_budget_counts_actual_state_storage_and_rebuild():
    """The budget must equal real tensor storage, including histories and sibling states."""
    group=LinearGatedDeltaGroupConfig(name='linear',layer_ids=(0,1),num_key_heads=2,
        num_value_heads=4,key_head_dim=8,value_head_dim=8,conv_kernel_dim=4,output_gate='sigmoid')
    model=SimpleNamespace(linear_attention_group=lambda:group,slot_states=(),
                          quant=SimpleNamespace(dialect='gguf_strata'))
    config=SimpleNamespace(model_config=model,tp_info=SimpleNamespace(size=1),dtype=torch.float32,
        cache_type='hybrid_radix',max_running_req=1,linear_state_cache_ratio=2.,linear_state_cache_slots=1,
        mtp_runtime_path='independent-mtp',mtp_compact_state=True)
    slots=_linear_pool_num_slots(config)
    pool=LinearStatePool(group,slots,torch.float32,torch.device('cpu'),tp_size=1,
                         reserved_slots=4,compact_mtp=True)
    for target in (slots,slots+1,slots):
        if target!=pool.num_slots:pool.rebuild(target)
        actual=sum(t.numel()*t.element_size() for t in [pool.conv_states,pool.recurrent_states,
            pool._compact_keys,pool._compact_delta,pool._compact_decay,pool._compact_source])
        assert state_pool_bytes(config,target)==actual
        assert not set(pool.alloc(pool.num_free_slots))&set(pool.reserved_slot_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_prefix_graph_and_rebuild():
    """Use real conv and recurrence kernels as the reference, not a copy of update-history logic."""
    torch.manual_seed(13579)
    device=torch.device('cuda')
    group=LinearGatedDeltaGroupConfig(name='linear',layer_ids=(0,1),num_key_heads=16,
        num_value_heads=48,key_head_dim=128,value_head_dim=128,conv_kernel_dim=4,output_gate='sigmoid')
    baseline=LinearStatePool(group,8,torch.float32,device,tp_size=1)
    pool=LinearStatePool(group,8,torch.float32,device,tp_size=1,reserved_slots=4,compact_mtp=True)
    set_global_ctx(Context(64,linear_state_pool=pool))
    ops=[]
    for layer in range(2):
        op=object.__new__(Qwen4ExpGatedDeltaNet)
        op.layer_id=layer
        op.num_k_heads,op.num_v_heads=16,48
        op.head_k_dim,op.head_v_dim=128,128
        op.key_dim,op.value_dim=2048,6144
        op.conv1d=SimpleNamespace(weight=torch.randn(10240,1,4,device=device))
        op.A_log=-torch.rand(48,device=device)
        op.dt_bias=torch.randn(48,device=device)
        ops.append(op)
    checks=[]
    slots=torch.tensor([1],device=device,dtype=torch.int32)
    for count in (1,2,3,4):
        pool.conv_states.normal_(0,.1)
        pool.recurrent_states.normal_(0,.01)
        initial_conv=pool.conv_states.clone()
        initial_rec=pool.recurrent_states.clone()
        conv=torch.randn(count,10240,device=device)
        a,b=torch.randn(count,48,device=device),torch.randn(count,48,device=device)
        fla=FLAMetadata(torch.tensor([0,count],device=device,dtype=torch.int32),slots,
            track_dst=torch.tensor(pool.reserved_slot_ids[:count],device=device),
            track_boundary_row=torch.arange(1,count+1,device=device),is_verify=True,compact_mtp_pool=pool)
        one=FLAMetadata(torch.tensor([0,1],device=device,dtype=torch.int32),slots)
        expected_states={}
        expected_conv={}
        expected_outputs=[]
        for layer,op in enumerate(ops):
            baseline.conv_states[layer].copy_(initial_conv[layer])
            baseline.recurrent_states[layer,:4].copy_(initial_rec[layer])
            outputs=[]
            for row in range(count):
                outputs.append(op._gguf_core(conv[row:row+1],a[row:row+1],b[row:row+1],baseline,layer,one,True))
                expected_states[layer,row]=baseline.recurrent_states[layer,1].clone()
                expected_conv[layer,row]=baseline.conv_states[layer,1].clone()
            expected_outputs.append(torch.cat(outputs))
            baseline.conv_states[layer].copy_(initial_conv[layer])
            baseline.recurrent_states[layer,:4].copy_(initial_rec[layer])
            legacy_verify=op._gguf_core(conv,a,b,baseline,layer,replace(fla,compact_mtp_pool=None),False)
            result=op._gguf_core(conv,a,b,pool,layer,fla,False)
            print(json.dumps(dict(count=count,layer=layer,
                legacy_verify_vs_native=int((legacy_verify!=expected_outputs[-1]).sum()),
                compact_vs_legacy_verify=int((result!=legacy_verify).sum()))),flush=True)
            torch.testing.assert_close(result,expected_outputs[-1],rtol=1e-6,atol=2e-9)
        torch.testing.assert_close(pool.recurrent_states,initial_rec,rtol=0,atol=0)
        for row in range(count):
            pool.copy_from(pool.reserved_slot_ids[row],3)
            for layer in range(2):
                print(json.dumps(dict(prefix=row+1,layer=layer,
                    state_nonidentical=int((pool.recurrent_states[layer,3]!=expected_states[layer,row]).sum()),
                    state_max_abs=float((pool.recurrent_states[layer,3]-expected_states[layer,row]).abs().max()))),flush=True)
                torch.testing.assert_close(pool.recurrent_states[layer,3],expected_states[layer,row],rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(pool.conv_states[layer,3],expected_conv[layer,row],rtol=0,atol=0)
        checks.append(f'{count}_rows_output_and_all_prefixes_fp32_reference_bound')
        if count==4:
            pool.conv_states.copy_(initial_conv)
            pool.recurrent_states.copy_(initial_rec)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                for layer,op in enumerate(ops):op._gguf_core(conv,a,b,pool,layer,fla,False)
                with torch.cuda.graph(graph,stream=stream):
                    captured=[op._gguf_core(conv,a,b,pool,layer,fla,False) for layer,op in enumerate(ops)]
            torch.cuda.current_stream().wait_stream(stream)
            for source in (2,1,2):
                pool.conv_states.copy_(initial_conv)
                pool.recurrent_states.copy_(initial_rec)
                conv.normal_();a.normal_();b.normal_();slots.fill_(source)
                graph.replay()
                for layer,op in enumerate(ops):
                    baseline.conv_states[layer].copy_(initial_conv[layer])
                    baseline.recurrent_states[layer,:4].copy_(initial_rec[layer])
                    outputs=[]
                    for row in range(4):
                        outputs.append(op._gguf_core(conv[row:row+1],a[row:row+1],b[row:row+1],baseline,layer,one,True))
                        expected_states[layer,row]=baseline.recurrent_states[layer,source].clone()
                    torch.testing.assert_close(captured[layer],torch.cat(outputs),rtol=1e-6,atol=2e-9)
                # Freeze the shorter anchor before advancing live, as the runtime does.
                pool.copy_from(pool.reserved_slot_ids[0],3)
                pool.copy_from(pool.reserved_slot_ids[3],source)
                for layer in range(2):
                    torch.testing.assert_close(pool.recurrent_states[layer,3],expected_states[layer,0],rtol=1e-6,atol=1e-7)
                    torch.testing.assert_close(pool.recurrent_states[layer,source],expected_states[layer,3],rtol=1e-6,atol=1e-7)
            graph.reset()
            checks.append('changed_input_and_request_slot_graph_replays_with_anchor_before_live')
    actual_bytes=pool.recurrent_states.numel()*4+sum(t.numel()*t.element_size() for t in
        (pool._compact_keys,pool._compact_delta,pool._compact_decay,pool._compact_source))
    old_bytes=2*8*48*128*128*4
    assert actual_bytes < old_bytes
    pool.rebuild(10)
    allocated=pool.alloc(pool.num_free_slots)
    assert set(allocated)==set(range(1,6)) and pool.recurrent_states.shape[1]==6
    assert not (set(allocated)&set(pool.reserved_slot_ids))
    pool.free(allocated)
    pool.recurrent_states.fill_(1)
    pool.clear_slots([2,*pool.reserved_slot_ids])
    assert not torch.count_nonzero(pool.recurrent_states[:,2])
    assert torch.all(pool.recurrent_states[:,1]==1)
    checks.append('rebuild_reserved_ownership_and_mixed_slot_clear')
    assert len(checks)==6
    assert old_bytes>actual_bytes
