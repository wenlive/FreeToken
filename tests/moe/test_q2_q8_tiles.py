"""Compare the wider vector tile with the frozen four-row kernel on real expert geometry."""
import pytest
import torch
import triton
from freetoken.moe import fused_q2_q8 as q2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('tokens',[1,2,3,4])
def test_wide_tile_routes_and_changed_graph_inputs(tokens):
    """Check both projections, poison unwritten output, then change values and route order."""
    torch.manual_seed(197+tokens)
    ids=torch.stack([torch.arange(10,device='cuda')+3*t for t in range(tokens)]).int()
    for k,n,input_topk in ((2560,1280,10),(640,2560,1)):
        bank=torch.randint(0,256,(24,n,k//64,18),device='cuda',dtype=torch.uint8)
        scale=(torch.rand((24,n,k//64),device='cuda')*.01+.001).half().contiguous()
        bank[...,:2]=scale.view(torch.uint8).reshape(24,n,k//64,2)
        bank=bank.flatten(2).contiguous()
        value=torch.randn((tokens*10//input_topk,k),device='cuda')
        result=torch.full((tokens*10,n),float('nan'),device='cuda')
        expected=torch.empty_like(result)

        def reference():
            """The old four-row kernel remains an independent launch/layout reference."""
            q,s=q2.quantize_expert_activation(value)
            q2._routed_vector[(ids.numel(),triton.cdiv(n,4))](q.view(torch.int32),s,bank,ids,expected,
                k,n,input_topk,4,triton.next_power_of_2(k//32),num_warps=4,enable_fp_fusion=False)

        q2._routed_project(value,bank,result,ids,input_topk)
        reference()
        torch.testing.assert_close(result,expected,rtol=1e-6,atol=2e-6)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):q2._routed_project(value,bank,result,ids,input_topk)
        for _ in range(3):
            value.normal_()
            ids.copy_(ids.roll(2,dims=1).roll(1,dims=0))
            result.fill_(float('nan'))
            graph.replay()
            reference()
            torch.testing.assert_close(result,expected,rtol=1e-6,atol=2e-6)
            assert torch.isfinite(result).all()
        graph.reset()
