"""真实 GPU 参考验证，覆盖专家归组、padding 和不同路由分布。"""
import pytest
import torch

from freetoken.moe.fused_q2_0 import grouped_experts_q2_0


def _weights(experts, n, k, device):
    """构造量化块和独立数学参考，避免以优化代码生成参考答案。"""
    scales = torch.rand((experts, n, k // 64), device=device).half() * .025
    codes = torch.randint(0, 4, (experts, n, k), device=device, dtype=torch.uint8)
    blocks = torch.empty((experts, n, k // 64, 18), dtype=torch.uint8, device=device)
    blocks[..., :2] = scales.contiguous().view(torch.uint8).reshape(experts, n, k // 64, 2)
    fields = codes.reshape(experts, n, k // 64, 16, 4)
    blocks[..., 2:] = fields[..., 0] | (fields[..., 1] << 2) | (fields[..., 2] << 4) | (fields[..., 3] << 6)
    decoded = ((codes.float() - 1) * scales.float().repeat_interleave(64, dim=-1)).bfloat16()
    return blocks.reshape(experts, n, k // 64 * 18), decoded


@pytest.mark.parametrize('tokens,topk,skew', [(1, 1, False), (19, 3, False), (129, 3, False), (257, 3, True)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_grouped_q2_matches_independent_bf16_reference(tokens, topk, skew):
    """跨 tile 边界、空专家和集中路由都必须保留原 Q2 量化与舍入语义。"""
    torch.manual_seed(321)
    experts, hidden, intermediate = 8, 128, 64
    gu, gu_ref = _weights(experts, 2 * intermediate, hidden, 'cuda')
    down, down_ref = _weights(experts, hidden, intermediate, 'cuda')
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device='cuda')
    ids = torch.randint(0, 2 if skew else experts, (tokens, topk), device='cuda', dtype=torch.int32)
    weights = torch.softmax(torch.randn((tokens, topk), device='cuda'), dim=-1)
    expected = torch.zeros((tokens, topk, hidden), device='cuda')
    # 每个专家一次参考 GEMM；参考不使用优化代码的归组/padding 或 Q2 解码函数。
    for expert in range(experts):
        token, route = torch.where(ids == expert)
        gu_values = (x[token].float() @ gu_ref[expert].float().T).bfloat16().float()
        gate, up = gu_values.chunk(2, dim=-1)
        active = (torch.nn.functional.silu(gate) * up).bfloat16().float()
        values = (active @ down_ref[expert].float().T).bfloat16().float()
        expected[token, route] = values * weights[token, route, None]
    expected = expected.sum(1).bfloat16()
    actual = grouped_experts_q2_0(x, gu, down, weights, ids)
    torch.testing.assert_close(actual, expected, atol=.01, rtol=.02)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all() and relative_l2 < .008
