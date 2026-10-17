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


def _integer_scaled_projection(x, packed):
    """Independent integer chunk dots, FP16 weight scales, FP32 activation scales."""
    n, k = packed.shape[0], packed.shape[1] // 18 * 64
    blocks = packed.reshape(n, k // 64, 18)
    dw = blocks[..., :2].contiguous().view(torch.float16).reshape(n, k // 64).float()
    fields = blocks[..., 2:, None].to(torch.int32)
    shifts = torch.arange(4, device=x.device) * 2
    w = (((fields >> shifts) & 3) - 1).reshape(n, k).float()
    xb = x.reshape(-1, 32)
    dx = xb.abs().amax(-1) / 127
    inverse = torch.where(dx > 0, dx.reciprocal(), 0)
    z = xb * inverse[:, None]
    q = torch.trunc(z + torch.where(z >= 0, .5, -.5)).clamp(-127, 127)
    result = torch.zeros(n, device=x.device, dtype=torch.float32)
    for group in range(k // 32):
        # These integer products/sums fit exactly in FP32, independent of TF32.
        dot = (w[:, group * 32:(group + 1) * 32] * q[group]).sum(-1)
        result += dot * dw[:, group // 2] * dx[group]
    return result


def _integer_expert_reference(x, gu, down, weights, ids):
    expected = torch.empty_like(x)
    for token in range(len(x)):
        parts = []
        for route, expert in enumerate(ids[token].tolist()):
            gate, up = _integer_scaled_projection(x[token], gu[expert]).chunk(2)
            active = torch.nn.functional.silu(gate) * up
            parts.append(_integer_scaled_projection(active, down[expert]) * weights[token, route])
        expected[token] = torch.stack(parts).sum(0)
    return expected


@pytest.mark.parametrize('tokens,prefill,experts', [(1, False, 16), (3, False, 16),
    (8, False, 16), (9, False, 16), (3, True, 16), (1, False, 4096)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_q2_q8_routed_experts_match_integer_reference(tokens, prefill, experts):
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    torch.manual_seed(932)
    hidden, intermediate, topk = 128, 64, 3
    gu, _ = _weights(experts, 2 * intermediate, hidden, 'cuda')
    down, _ = _weights(experts, hidden, intermediate, 'cuda')
    x = torch.randn((tokens, hidden), device='cuda', dtype=torch.float32)
    # Sparse high slot ids and duplicated routes must not confuse source identity.
    ids = torch.tensor([experts - 1, 7, 7], device='cuda', dtype=torch.int32)[None].repeat(tokens, 1)
    weights = torch.softmax(torch.randn((tokens, topk), device='cuda'), -1)
    expected = _integer_expert_reference(x, gu, down, weights, ids)
    actual = fused_experts_q2_q8(x, gu, down, weights, ids, is_prefill=prefill)
    assert torch.isfinite(actual).all()
    assert (actual - expected).norm() / expected.norm().clamp_min(1e-10) < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_q2_q8_decode_skips_full_cache_grouping_and_replays_graph(monkeypatch):
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    import freetoken.moe.fused as fused
    def forbidden(*args):
        raise AssertionError('Sparse decode must not scan all cache slots')
    monkeypatch.setattr(fused, 'moe_align_block_size', forbidden)
    torch.manual_seed(832)
    gu, _ = _weights(16, 128, 128, 'cuda')
    down, _ = _weights(16, 128, 64, 'cuda')
    x = torch.randn((1, 128), device='cuda', dtype=torch.float32)
    ids = torch.tensor([[15, 7, 7]], device='cuda', dtype=torch.int32)
    weights = torch.tensor([[.2, .3, .5]], device='cuda', dtype=torch.float32)
    for _ in range(3):fused_experts_q2_q8(x, gu, down, weights, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual = fused_experts_q2_q8(x, gu, down, weights, ids)
    x.mul_(1.3)
    ids[:, 0] = 11
    graph.replay()
    torch.cuda.synchronize()
    expected = _integer_expert_reference(x, gu, down, weights, ids)
    assert torch.isfinite(actual).all()
    assert (actual - expected).norm() / expected.norm().clamp_min(1e-10) < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_q2_q8_zero_empty_and_invalid_precision():
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    gu, _ = _weights(4, 128, 128, 'cuda')
    down, _ = _weights(4, 128, 64, 'cuda')
    x = torch.zeros((1, 128), device='cuda', dtype=torch.float32)
    ids = torch.tensor([[3]], device='cuda', dtype=torch.int64)
    weights = torch.ones((1, 1), device='cuda', dtype=torch.float32)
    actual = fused_experts_q2_q8(x, gu, down, weights, ids)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) == 0
    assert fused_experts_q2_q8(x[:0], gu, down, weights[:0], ids[:0]).shape == (0, 128)
    with pytest.raises(ValueError, match='FP32 routing'):
        fused_experts_q2_q8(x, gu, down, weights.half(), ids)
