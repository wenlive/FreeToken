import pytest
import torch

from freetoken.lora.manifest import LoRAPair
from freetoken.lora.ops import expert_residual, linear_residual


def test_fused_segments_preserve_other_columns():
    torch.manual_seed(19)
    x, y = torch.randn(5, 7), torch.randn(5, 15).bfloat16()
    a, b = torch.randn(2, 7), torch.randn(6, 2)
    out = linear_residual(x, y, [(3, 9, LoRAPair(a, b, 2.0))])
    expected = y.float()
    expected[:, 3:9] += (x @ a.T @ b.T) * 2
    assert torch.equal(out, expected.bfloat16())
    assert torch.equal(out[:, :3], y[:, :3]) and torch.equal(out[:, 9:], y[:, 9:])


@pytest.mark.parametrize("routes", [2, 300])
@pytest.mark.parametrize("routed", [True, False])
def test_expert_logical_ids_and_chunk_boundary(routes, routed):
    torch.manual_seed(21)
    ids = torch.randint(0, 11, (routes, 3), dtype=torch.int32)
    x = torch.randn(routes * 3 if routed else routes, 7).bfloat16()
    pair = LoRAPair(torch.randn(11, 2, 7), torch.randn(11, 9, 2), 2.0)
    expected = []
    for route, expert in enumerate(ids.flatten().tolist()):
        inp = x[route if routed else route // 3].float()
        expected.append((pair.b[expert] @ (pair.a[expert] @ inp)) * 2)
    torch.testing.assert_close(expert_residual(x, ids, pair, routed=routed), torch.stack(expected), rtol=2e-6, atol=2e-5)
