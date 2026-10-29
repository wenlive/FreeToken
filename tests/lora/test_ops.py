import pytest
import torch

from freetoken.lora.manifest import LoRAPair
from freetoken.lora.ops import expert_residual, linear_residual, pack_linear


def test_fused_segments_preserve_other_columns():
    torch.manual_seed(19)
    x, y = torch.randn(5, 7), torch.randn(5, 15).bfloat16()
    a, b = torch.randn(2, 7), torch.randn(6, 2)
    out = linear_residual(x, y, pack_linear([(3, 9, LoRAPair(a, b, 2.0))], 7, 15, "cpu"))
    expected = y.float()
    expected[:, 3:9] += (x @ a.T @ b.T) * 2
    assert torch.equal(out, expected.bfloat16())
    assert torch.equal(out[:, :3], y[:, :3]) and torch.equal(out[:, 9:], y[:, 9:])


@pytest.mark.parametrize("rows", [1, 4, 19])
def test_compiled_linear_matches_separate_segments(rows):
    torch.manual_seed(79)
    x, y = torch.randn(rows, 17).bfloat16(), torch.randn(rows, 31).bfloat16()
    parts = [(0, 11, LoRAPair(torch.randn(3, 17), torch.randn(11, 3), 2.0)),
             (11, 23, LoRAPair(torch.randn(5, 17), torch.randn(12, 5), 0.5))]
    expected = y.float()
    for start, end, pair in parts:
        expected[:, start:end] += (x.float() @ pair.a.T @ pair.b.T) * pair.scaling
    packed = pack_linear(parts, 17, 31, "cpu")
    torch.testing.assert_close(linear_residual(x, y, packed), expected.bfloat16(), rtol=.008, atol=.008)
    assert torch.equal(linear_residual(x, y, packed)[:, 23:], y[:, 23:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 17])
def test_cuda_packed_linear_three_segments_and_graph(rows):
    """Compare FP32 low-rank math, untouched columns and captured execution."""
    torch.manual_seed(81)
    torch.backends.cuda.matmul.allow_tf32 = False
    x, y = torch.randn(rows, 2048).bfloat16(), torch.randn(rows, 97).bfloat16()
    parts = [(start, end, LoRAPair(torch.randn(16, 2048) * .02, torch.randn(end-start, 16) * .02, 2.0))
             for start, end in [(0, 64), (64, 80), (80, 96)]]
    reference = y.float()
    for start, end, pair in parts:
        reference[:, start:end] += x.float() @ pair.a.T @ pair.b.T * pair.scaling
    gx, gy = x.cuda(), y.cuda()
    packed = pack_linear(parts, 2048, 97, "cuda")
    output = linear_residual(gx, gy, packed)
    torch.testing.assert_close(output.cpu(), reference.bfloat16(), rtol=.008, atol=.008)
    assert torch.equal(output[:, 96:], gy[:, 96:])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        linear_residual(gx, gy, packed)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = linear_residual(gx, gy, packed)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, captured)


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
