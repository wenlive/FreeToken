"""Exercise owner lifetime, physical bytes and graph-safe mixed-source admission."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.moe.base import BankSpec
from freetoken.moe.q2_residency import ProtectedQ2Cache, load_compact_banks, protected_ranking


def test_config_rejects_unsupported_contract_before_loading():
    kwargs = dict(model_path='unused', tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
                  moe_strategy='offload', max_running_req=1, moe_cache_size=11264,
                  moe_q2_resident_experts=9216)
    EngineConfig(**kwargs)
    for dynamic in (0,512,1024):
        EngineConfig(**(kwargs | dict(moe_q2_dynamic_slots=dynamic,moe_cache_size=1024+9216+dynamic)))
    with pytest.raises(ValueError,match='nonnegative'):
        EngineConfig(**(kwargs | dict(moe_q2_dynamic_slots=-1)))
    for override in [dict(moe_cache_size=11263), dict(moe_cache_auto=True),
                     dict(max_running_req=2), dict(moe_cpu_layers='auto'),
                     dict(moe_strategy='hybrid'), dict(moe_cache_policy='lfu')]:
        with pytest.raises(ValueError, match='Protected Q2'):
            EngineConfig(**(kwargs | override))
    with pytest.raises(ValueError, match='geometry'):
        protected_ranking(48, 511, 9216)


def _make_cache(valid_scales=False,dynamic_slots=None):
    layers, experts, hidden, intermediate = 3, 16, 128, 64
    layout = {'gate_up': BankSpec((2 * intermediate, hidden // 64 * 18), torch.uint8),
              'down': BankSpec((hidden, intermediate // 64 * 18), torch.uint8)}
    original = {role: [torch.randint(0, 256, (experts, *spec.shape), dtype=torch.uint8)
                       for _ in range(layers)] for role, spec in layout.items()}
    if valid_scales:
        scale = torch.tensor([.015625], dtype=torch.float16).view(torch.uint8)
        for banks in original.values():
            for bank in banks:
                bank.view(*bank.shape[:-1], -1, 18)[..., :2] = scale
    pieces = [(layer, e, min(e + 3, experts),
               {role: per_layer[layer][e:min(e + 3, experts)] for role, per_layer in original.items()})
              for layer in range(layers) for e in range(0, experts, 3)]

    def pack(piece, out):
        for role in layout:
            out[role].copy_(piece[role])
        return {}

    protected = [(0, e) for e in range(experts)] + [(1, e) for e in (0, 3, 6, 9)]
    method = SimpleNamespace(kind=QuantKind.Q2_0, kernel=SimpleNamespace(name='triton'),
                             layout=lambda: layout, pack=pack)
    config = SimpleNamespace(num_moe_layers=layers, num_experts=experts)
    with patch('freetoken.moe.expert_pieces.iter_expert_pieces', return_value=iter(pieces)):
        banks, ownership = load_compact_banks('deterministic-fixture', config,
            protected=protected, dynamic=experts if dynamic_slots is None else dynamic_slots, method=method,
            device=torch.device('cuda'), dtype=torch.bfloat16)
    cache = ProtectedQ2Cache(layers, experts, ownership.slots, torch.device('cuda'),
        ownership=ownership, prefill_overlap=True, quant_format='q2_0', layout=layout)
    cache.set_bank_sources(banks.sources)
    return cache, original


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dynamic_slots',[0,16])
def test_decode_copy_overlap_graph_prefill_reset_and_changed_routes(dynamic_slots):
    """真实Q2计算覆盖跨流依赖、换路由、prefill穿插和reset，不用host同步掩盖遗漏事件。"""
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.fused import fused_topk
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    cache, original = _make_cache(valid_scales=True,dynamic_slots=dynamic_slots)
    x = torch.randn(1, 128, device='cuda')
    logits = torch.randn(1, 16, device='cuda')
    shared_weight = torch.randn(128, 128, device='cuda')
    full = {role: banks[1].to('cuda') for role, banks in original.items()}

    def gemm(cache, value, weights, ids, **kwargs):
        return fused_experts_q2_q8(value, cache.bank_caches['gate_up'],
                                   cache.bank_caches['down'], weights, ids)

    layer = SimpleNamespace(offload_cache=cache, top_k=3, renormalize=True, layer_id=1,
                            _expert_gemm=gemm, _maybe_all_reduce=lambda value: value)
    def shared(value):
        return value @ shared_weight
    def forward():
        return OffloadMoELayer.decode_with_shared(layer, x, logits, shared)
    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed, independent = forward()
    for cycle in range(6):
        cache.reset()
        if cycle % 2:
            cache.begin_prefill()
            for index in range(3):
                cache.prefetch_prefill_layer(index)
                cache.wait_prefill_layer(index)
                cache.release_prefill_layer(index)
        x.normal_()
        logits.normal_()
        graph.replay()
        weights, ids = fused_topk(hidden_states=x, gating_output=logits, topk=3, renormalize=True)
        reference = fused_experts_q2_q8(x, full['gate_up'], full['down'], weights, ids)
        torch.testing.assert_close(routed, reference, rtol=0, atol=0)
        torch.testing.assert_close(independent, x @ shared_weight, rtol=0, atol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('tokens,topk', [(1, 3), (3, 3), (8, 3), (16, 10)])
def test_owner_bytes_survive_graph_reset_prefill_and_rebuild(tokens, topk):
    torch.manual_seed(7801)
    cache, original = _make_cache()
    p = cache.protected_payload
    saved = {role: value[p.protected_begin:].clone() for role, value in cache.bank_caches.items()}
    addresses = {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    assert all(cache.bank_sources[r][0].shape[0] == 0 for r in original)

    def verify(layer, logical, physical):
        torch.cuda.synchronize()
        for role in original:
            assert torch.equal(cache.bank_caches[role][physical.long()].cpu(), original[role][layer][logical.long()])
            assert torch.equal(cache.bank_caches[role][p.protected_begin:], saved[role])

    ids = torch.empty((tokens, topk), dtype=torch.int32, device='cuda')
    initial = torch.randint(0, 16, ids.shape, device='cuda', dtype=torch.int32)
    ids.copy_(initial)
    cache.ensure_experts(1, ids)
    cache.copy_missing()
    verify(1, initial.cpu(), ids)
    ids.copy_(initial)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.ensure_experts(1, ids)
        cache.copy_missing()
    for cycle in range(8):
        cache.reset()
        logical = torch.randint(0, 16, ids.shape, device='cuda', dtype=torch.int32)
        logical[:, 2] = logical[:, 1]
        ids.copy_(logical)
        graph.replay()
        verify(1, logical.cpu(), ids)
        cache.begin_prefill()
        for layer in range(3):
            cache.prefetch_prefill_layer(layer)
            views = cache.wait_prefill_layer(layer)
            for role, view in zip(original, views):
                assert torch.equal(view.cpu(), original[role][layer])
            cache.release_prefill_layer(layer)
    assert torch.count_nonzero(cache.lru_stats) == 0  # Disabled counters are not captured.
    before = cache.slot_for_id.clone()
    with pytest.raises(ValueError, match='fixed'):
        cache.rebuild(cache.cache_size + 1)
    assert torch.equal(before, cache.slot_for_id)
    cache.rebuild(cache.cache_size)
    assert addresses == {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    ids.copy_(initial)
    graph.replay()
    verify(1, initial.cpu(), ids)
    cache.materialize_layer(2)
    cache.copy_missing()
    for role, view in zip(original, cache.bank_views(16)):
        assert torch.equal(view.cpu(), original[role][2])
    # A second instance owns its own tables and data; no module-level loader state.
    second, second_original = _make_cache()
    other_ids = torch.tensor([[15, 7, 7]], device='cuda', dtype=torch.int32)
    second.collect_stats = True
    second.ensure_experts(2, other_ids)
    second.copy_missing()
    for role in original:
        assert torch.equal(second.bank_caches[role][other_ids.long()].cpu(), second_original[role][2][[15, 7, 7]][None])
        assert torch.equal(cache.bank_caches[role][p.protected_begin:], saved[role])
    assert second.lru_stats[2].tolist() == [2, 2, 1]


@pytest.mark.parametrize('tokens',[1,4])
def test_borrowed_prefill_bytes_are_invalidated_before_next_decode(tokens):
    """不在阶段转换前做host同步；同一个CUDA图在prefill后必须重新取到正确专家。"""
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    from freetoken.moe.fused import fused_topk
    cache,original=_make_cache(valid_scales=True)
    full={role:rows[2].to('cuda') for role,rows in original.items()}
    permanent={role:bank[cache.protected_payload.protected_begin:].clone() for role,bank in cache.bank_caches.items()}
    logical=torch.arange(16,device='cuda',dtype=torch.int32).view(4,4)
    slots=logical.clone();cache.ensure_experts(2,slots);cache.copy_missing()
    assert torch.all(slots<32)
    x=torch.randn(tokens,128,device='cuda');logits=torch.randn(tokens,16,device='cuda');shared_weight=torch.randn(128,128,device='cuda')
    def gemm(cache,value,weights,ids,**kw):return fused_experts_q2_q8(value,cache.bank_caches['gate_up'],cache.bank_caches['down'],weights,ids)
    layer=SimpleNamespace(offload_cache=cache,top_k=4,renormalize=True,layer_id=2,_expert_gemm=gemm,_maybe_all_reduce=lambda x:x)
    def forward():return OffloadMoELayer.decode_with_shared(layer,x,logits,lambda v:v@shared_weight)
    forward();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):routed,shared=forward()
    for cycle in range(8):
        x.normal_();logits.normal_();graph.replay()
        before=routed.clone()
        w,ids=fused_topk(hidden_states=x,gating_output=logits,topk=4,renormalize=True)
        reference=fused_experts_q2_q8(x,full['gate_up'],full['down'],w,ids)
        cache.begin_prefill()
        for index in range(3):
            cache.prefetch_prefill_layer(index);cache.wait_prefill_layer(index);cache.release_prefill_layer(index)
        graph.replay()
        torch.testing.assert_close(before,reference,rtol=0,atol=0)
        torch.testing.assert_close(routed,reference,rtol=0,atol=0)
        torch.testing.assert_close(shared,x@shared_weight,rtol=0,atol=0)
        for role in original:torch.testing.assert_close(cache.bank_caches[role][cache.protected_payload.protected_begin:],permanent[role],rtol=0,atol=0)
    graph.reset();cache.reset()
