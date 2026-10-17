"""Check exchanged owner bytes, captured graph replay and full-layer prefill on CUDA."""
from pathlib import Path
import pytest
import runpy
import torch
from freetoken.moe.q2_adaptive import AdaptiveQ2Owners


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize('dynamic_slots',[0,16])
def test_online_exchange_keeps_all_owner_bytes_and_graphs(dynamic_slots):
    """Compare every expert with original bytes after exchanges in both directions."""
    _make_cache=runpy.run_path(str(Path(__file__).with_name('test_q2_residency.py')))['_make_cache']
    cache,original=_make_cache(valid_scales=True,dynamic_slots=dynamic_slots)
    controller=AdaptiveQ2Owners(cache,max_swaps=2)
    payload=cache.protected_payload
    initial_addresses={r:t.data_ptr() for r,t in cache.bank_caches.items()}
    ids=torch.tensor([[2,5,0]],device='cuda',dtype=torch.int32)
    logical=ids.cpu().clone()
    cache.ensure_experts(1,ids)
    cache.copy_missing()
    ids.copy_(logical)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.ensure_experts(1,ids)
        cache.copy_missing()
    checks=[]
    for cycle in range(4):
        cache.decode_freq.zero_()
        controller.last_counts.fill(0)
        controller.heat.fill(0)
        targets=(2,5) if cycle%2==0 else (0,3)
        cache.decode_freq[1,targets[0]]=100
        cache.decode_freq[1,targets[1]]=80
        controller.adapt()
        assert all((1,e) in payload.protected_slots for e in targets)
        for reset in (False,True):
            if reset:cache.reset()
            ids.copy_(logical)
            graph.replay()
            torch.cuda.synchronize()
            for role in original:
                actual=cache.bank_caches[role][ids.long()].cpu()
                expected=original[role][1][logical.long()]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Include GPU-only, mixed-owner and host-only layers in the prefill check.
        cache.begin_prefill()
        for layer in range(3):
            cache.prefetch_prefill_layer(layer)
            views=cache.wait_prefill_layer(layer)
            for role,view in zip(original,views):
                torch.testing.assert_close(view.cpu(),original[role][layer],rtol=0,atol=0)
            cache.release_prefill_layer(layer)
        checks.append(f'bidirectional_swap_{cycle}_graph_reset_and_prefill_all_expert_bytes')
    assert initial_addresses=={r:t.data_ptr() for r,t in cache.bank_caches.items()}
    assert len(payload.protected_slots)==20
    for layer in range(payload.layers):
        host=[i for i in payload.host_indices[layer] if i>=0]
        assert len(set(host))==len(host)
        assert sorted(host)==list(range(len(host)))
    graph.reset()
    assert len(checks)==4
    assert controller.staging_bytes==0
    assert sum(row['swaps'] for row in controller.rounds)==8
