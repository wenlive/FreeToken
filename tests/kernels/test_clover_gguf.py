"""Independent Q8_1 dot arithmetic and Clover's routed rounding boundaries."""

from pathlib import Path
import os

import gguf
import numpy as np
import pytest
import torch


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    import clover_gguf_reference

    return clover_gguf_reference


def test_q8_1_rounding_is_per_block_away_from_zero_and_zero_safe(reference):
    value = torch.zeros(1, 64)
    value[0, :5] = torch.tensor([127.0, 0.5, -0.5, 1.5, -1.5])
    result = reference.q8_1_activations(value)
    assert result[0, :5].tolist() == [127.0, 1.0, -1.0, 2.0, -2.0]
    assert torch.equal(result[:, 32:], torch.zeros(1, 32))


def test_clover_keeps_down_and_router_fp32_until_routed_rounding(monkeypatch):
    from freetoken.moe import fused_gguf as module

    hidden = torch.ones(1, 2, dtype=torch.bfloat16)
    route_weights = torch.tensor([[0.1234567, 0.8765433]], dtype=torch.float32)
    precise_down = torch.tensor([[1.0031, -1.0049], [0.9981, 1.0111]])
    calls = []

    def gemm(x, weight, ids, top_k, kind, rows, tokens):
        calls.append(x.dtype)
        if len(calls) == 1:
            return torch.zeros(2, 2, dtype=x.dtype)
        return precise_down.to(x.dtype)

    monkeypatch.setattr(module, "ggml_moe_a8_vec", gemm)
    monkeypatch.setattr(
        module, "silu_and_mul", lambda x: torch.ones(2, 1, dtype=x.dtype)
    )
    monkeypatch.setattr(
        module, "moe_sum_reduce_triton", lambda x, out: out.copy_(x.float().sum(1))
    )
    result = module.fused_experts_gguf(
        hidden,
        torch.zeros(2, 2, 1, dtype=torch.uint8),
        torch.zeros(2, 2, 1, dtype=torch.uint8),
        route_weights,
        torch.tensor([[0, 1]], dtype=torch.int32),
        "silu",
        gate_up_type=23,
        down_type=12,
    )
    expected = (
        (precise_down.reshape(1, 2, 2) * route_weights[..., None])
        .bfloat16()
        .float()
        .sum(1)
        .bfloat16()
    )
    early = (
        precise_down.bfloat16().reshape(1, 2, 2) * route_weights.bfloat16()[..., None]
    ).sum(1)
    assert calls == [torch.bfloat16, torch.float32]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert not torch.equal(result, early)


def packed_fixture(kind, experts=3, rows=64, width=256):
    rng = np.random.default_rng(100 + kind)
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
    raw = rng.integers(0, 256, (experts, rows, width // block, size), dtype=np.uint8)
    raw[..., :2] = np.array([0.0125], dtype=np.float16).view(np.uint8)
    if kind == 12:
        raw[..., 2:4] = np.array([0.009], dtype=np.float16).view(np.uint8)
    flat = raw.reshape(experts, rows, -1)
    dense = gguf.dequantize(flat, gguf.GGMLQuantizationType(kind))
    return torch.from_numpy(flat.copy()), torch.from_numpy(dense.copy())


def test_reference_cpu_cache_is_bounded_and_avoids_repeated_decode(
    reference, monkeypatch
):
    from types import SimpleNamespace

    packed, expected = packed_fixture(12, rows=2)
    tensor = SimpleNamespace(
        name="blk.0.ffn_down_exps.weight",
        shape=(3, 2, 256),
        _raw=packed.numpy(),
        ggml_type=12,
    )
    monkeypatch.setattr(reference, "iter_gguf_tensors", lambda _: [tensor])
    storage = reference.GGUFExpertStorage(
        "synthetic", device="cpu", cache_bytes=0, cpu_cache_bytes=4096
    )
    for expert in (0, 1, 0):
        torch.testing.assert_close(
            storage.projection(0, expert, "down"), expected[expert], rtol=0, atol=0
        )
    assert storage.cpu_hits == 1 and storage.cpu_bytes == 4096
    storage.projection(0, 2, "down")
    assert list(storage.cpu_cache) == [(0, 0, "down"), (0, 2, "down")]
    assert storage.cpu_bytes <= storage.cpu_limit and storage.bytes == 0


@pytest.mark.parametrize("kind", [12, 23])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eager_torch_decoder_matches_independent_numpy_bitwise(reference, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from clover_gguf_decode import decode_expert

    raw, expected = packed_fixture(kind, rows=19, width=1024)
    for expert in (0, 1, 2):
        actual = decode_expert(raw[expert].numpy(), kind, device=device)
        torch.testing.assert_close(actual.cpu(), expected[expert], rtol=0, atol=0)


@pytest.fixture(scope="module")
def real_gguf_experts():
    location = os.environ.get("CLOVER_GGUF_CHECKPOINT")
    if not location:
        pytest.skip("set CLOVER_GGUF_CHECKPOINT to the mixed Clover GGUF")
    from freetoken.models.gguf.reader import iter_gguf_tensors

    return {
        tensor.name: tensor
        for tensor in iter_gguf_tensors(location)
        if "_exps.weight" in tensor.name
    }


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layer", [0, 36])
@pytest.mark.parametrize("expert", [0, 127, 255])
@pytest.mark.parametrize("role", ["gate", "up", "down"])
def test_real_expert_torch_decoder_is_bitwise_numpy(
    reference, real_gguf_experts, layer, expert, role
):
    from clover_gguf_decode import decode_expert

    tensor = real_gguf_experts[f"blk.{layer}.ffn_{role}_exps.weight"]
    raw = tensor._raw.reshape(*tensor.shape[:2], -1)[expert]
    expected = torch.from_numpy(
        gguf.dequantize(raw, gguf.GGMLQuantizationType(tensor.ggml_type))
    )
    actual = decode_expert(raw, tensor.ggml_type, device="cuda")
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", [2, 8, 12, 23])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 8])
def test_native_grouped_gguf_dot_with_nonzero_slots_matches_independent_reference(
    reference, kind, dtype, width
):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    packed, dense = packed_fixture(kind)
    packed, dense = packed.cuda(), dense.cuda()
    generator = torch.Generator(device="cuda").manual_seed(177)
    value = torch.randn(width, 256, device="cuda", dtype=dtype, generator=generator)
    ids = torch.tensor([[2, 0]] * width, device="cuda", dtype=torch.int32)
    actual = ggml_moe_a8_vec(value, packed, ids, 2, kind, 64, width).reshape(
        width, 2, 64
    )
    activation = reference.q8_1_activations(value)
    expected = torch.stack(
        [
            torch.stack(
                [torch.mv(dense[int(expert)], activation[t]) for expert in ids[t]]
            )
            for t in range(width)
        ]
    )
    if kind == 2:
        # Q4_0 uses Q8_1's FP16 original-sum header for its -8 offset correction.
        sums = value.float().reshape(width, -1, 32).sum(-1).half().float()
        quant_sums = activation.reshape(width, -1, 32).sum(-1)
        scales = (
            packed.reshape(3, 64, -1, 18)[..., :2]
            .contiguous()
            .view(torch.float16)
            .float()
            .squeeze(-1)
        )
        for token in range(width):
            for slot in range(2):
                expected[token, slot] += -8.0 * (
                    scales[int(ids[token, slot])] * (sums[token] - quant_sums[token])
                ).sum(-1)
    expected = expected.to(dtype)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    assert error < (0.003 if dtype == torch.bfloat16 else 0.001), float(error)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 8, 16, 32])
def test_mixed_swiglu_routing_matches_independent_q8_reference(reference, width):
    from freetoken.moe.fused_gguf import fused_experts_gguf

    gate_up, dense_up = packed_fixture(23, rows=1024, width=2048)
    down, dense_down = packed_fixture(12, rows=2048, width=512)
    gate_up, down, dense_up, dense_down = [
        tensor.cuda() for tensor in (gate_up, down, dense_up, dense_down)
    ]
    generator = torch.Generator(device="cuda").manual_seed(198)
    value = torch.randn(
        width, 2048, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    ids = torch.tensor([[2, 0]] * width, dtype=torch.int32, device="cuda")
    weights = torch.tensor([[0.1234567, 0.8765433]] * width, device="cuda")
    actual = fused_experts_gguf(
        value, gate_up, down, weights, ids, "silu", gate_up_type=23, down_type=12
    )
    contributions = []
    for slot, expert in enumerate((2, 0)):
        projected = torch.nn.functional.linear(
            reference.q8_1_activations(value), dense_up[expert]
        ).bfloat16()
        gate, up = projected.chunk(2, -1)
        intermediate = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
        output = torch.nn.functional.linear(
            reference.q8_1_activations(intermediate.float()), dense_down[expert]
        )
        contributions.append((output * weights[:, slot, None]).bfloat16().float())
    expected = torch.stack(contributions).sum(0).bfloat16()
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    assert error < 0.01, float(error)
