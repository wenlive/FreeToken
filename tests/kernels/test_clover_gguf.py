"""Independent Q8_1 dot arithmetic and Clover's routed rounding boundaries."""

import os

import gguf
import numpy as np
import pytest
import torch


@pytest.fixture
def reference():
    from benchmarks.clover import clover_gguf_reference

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
    if kind == 14:
        raw[..., -2:] = np.array([0.0125], dtype=np.float16).view(np.uint8)
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


@pytest.mark.parametrize("kind", [2, 8, 12, 14, 23])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eager_torch_decoder_matches_independent_numpy_bitwise(reference, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from benchmarks.clover.clover_gguf_decode import decode_expert

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
    from benchmarks.clover.clover_gguf_decode import decode_expert

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
@pytest.mark.parametrize("kinds", [(23, 12), (8, 8), (12, 14)])
def test_mixed_swiglu_routing_matches_independent_q8_reference(reference, width, kinds):
    from freetoken.moe.fused_gguf import fused_experts_gguf

    gate_kind, down_kind = kinds
    gate_up, dense_up = packed_fixture(gate_kind, rows=1024, width=2048)
    down, dense_down = packed_fixture(down_kind, rows=2048, width=512)
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
        value, gate_up, down, weights, ids, "silu", gate_up_type=gate_kind, down_type=down_kind
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", [2, 8, 12, 14])
@pytest.mark.parametrize("width", [8, 16, 32])
def test_mmq_matches_independent_dense_q8_dot(reference, kind, width):
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.kernel.triton.clover_align import align_clover_routes

    packed, dense = packed_fixture(kind, rows=64, width=512)
    value = torch.randn(width, 512, device="cuda", dtype=torch.float32)
    ids = torch.tensor([[2, 0, 2]] * width, device="cuda", dtype=torch.int32)
    order, experts, total = align_clover_routes(ids, ggml_moe_get_block_size(kind), 3)
    actual = ggml_moe_a8(value, packed.cuda(), order, experts, total, kind, 64, 3, width)
    if kind == 2:
        # MMVQ and MMQ use the same independent Q8_1 original-sum correction;
        # the separately qualified vector path is an additional shape oracle.
        from freetoken.kernel.gguf import ggml_moe_a8_vec
        expected = ggml_moe_a8_vec(value, packed.cuda(), ids, 3, kind, 64, width)
    else:
        dense = dense.cuda()
        activation = reference.q8_1_activations(value)
        expected = torch.stack([torch.nn.functional.linear(activation, dense[e]) for e in (2, 0, 2)], 1).reshape(-1, 64)
    relative = (actual - expected).norm() / expected.norm()
    assert relative < 0.001, float(relative)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_short_route_grouping_replays_with_different_high_cache_slots():
    from freetoken.kernel.triton.clover_align import align_clover_routes

    ids = torch.tensor([[2999, 17, 2999], [2100, 17, 17]], device="cuda", dtype=torch.int32)
    align_clover_routes(ids, 4, 3000)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        order, experts, total = align_clover_routes(ids, 4, 3000)
    for routes in ([[2999, 17, 2999], [2100, 17, 17]], [[2, 2, 2], [2999, 2, 2999]]):
        ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
        graph.replay()
        n = int(total.cpu())
        sorted_ids = order[:n].cpu().tolist()
        owners = experts[:n // 4].cpu().tolist()
        assert sorted(i for i in sorted_ids if i < ids.numel()) == list(range(ids.numel()))
        flattened = [e for row in routes for e in row]
        for block, owner in enumerate(owners):
            assert all(flattened[i] == owner for i in sorted_ids[block * 4:(block + 1) * 4] if i < ids.numel())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", [2, 8, 12, 14, 23])
def test_mmvq_honors_padded_expert_slot_stride(kind):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    packed, _ = packed_fixture(kind, rows=64, width=512)
    packed = packed.cuda()
    padded = torch.zeros(3, packed[0].numel() + 128, device="cuda", dtype=torch.uint8)
    padded[:, :packed[0].numel()].copy_(packed.reshape(3, -1))
    view = padded.as_strided(packed.shape, (padded.stride(0), packed.stride(1), 1))
    value = torch.randn(1, 512, device="cuda")
    ids = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
    expected = ggml_moe_a8_vec(value, packed, ids, 2, kind, 64, 1)
    actual = ggml_moe_a8_vec(value, view, ids, 2, kind, 64, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 8])
def test_mixed_layers_share_q8_sized_banks_without_changing_results(rows):
    from freetoken.moe.fused_gguf import fused_experts_gguf

    gate_bank = torch.zeros(6, 512, 272, dtype=torch.uint8, device="cuda")
    down_bank = torch.zeros(6, 256, 272, dtype=torch.uint8, device="cuda")
    hidden = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
    weights = torch.full((rows, 2), .5, device="cuda")
    ids = torch.tensor([[2, 0]] * rows, dtype=torch.int32, device="cuda")
    for layer, (gate_kind, down_kind) in enumerate(((8, 8), (23, 12))):
        gate, _ = packed_fixture(gate_kind, rows=512, width=256)
        down, _ = packed_fixture(down_kind, rows=256, width=256)
        views = []
        for bank, packed in ((gate_bank, gate), (down_bank, down)):
            view = bank.as_strided((6, *packed.shape[1:]), (bank.stride(0), packed.stride(1), 1))
            view[layer * 3:layer * 3 + 3].copy_(packed)
            views.append(view)
        expected = fused_experts_gguf(hidden, gate.cuda(), down.cuda(), weights, ids,
            "silu", gate_up_type=gate_kind, down_type=down_kind)
        actual = fused_experts_gguf(hidden, *views, weights, ids + layer * 3,
            "silu", gate_up_type=gate_kind, down_type=down_kind)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 8, 16, 32, 1024])
def test_packed_dense_q8_mmvq_and_mmq_match_independent_dot(reference, width):
    from freetoken.models.clover_moe.packed import CloverPackedLinear

    packed, dense = packed_fixture(8, rows=128, width=512)
    layer = CloverPackedLinear(128, 512, 8, 544)
    layer.weight = packed[2].cuda()
    value = torch.randn(width, 512, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(reference.q8_1_activations(value), dense[2].cuda()).bfloat16()
    actual = layer.forward(value)
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert layer.weight.dtype == torch.uint8
    assert relative < .003, float(relative)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("stride", [34816, 4 << 20])
def test_mmq_large_cache_slots_and_byte_offsets_above_2gib(reference, stride):
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.kernel.triton.clover_align import align_clover_routes

    packed, dense = packed_fixture(8, rows=64, width=512)
    storage = torch.empty(513, stride, dtype=torch.uint8, device="cuda")
    slots = (0, 256, 512)
    for expert, slot in enumerate(slots):
        storage[slot, :packed[expert].numel()].copy_(packed[expert].reshape(-1))
    weights = storage.as_strided((513, 64, 544), (stride, 544, 1))
    value = torch.randn(8, 512, device="cuda", dtype=torch.float32)
    ids = torch.tensor([list(slots)] * 8, device="cuda", dtype=torch.int32)
    order, experts, total = align_clover_routes(ids, ggml_moe_get_block_size(8), 513)
    actual = ggml_moe_a8(value, weights, order, experts, total, 8, 64, 3, 8)
    activation = reference.q8_1_activations(value)
    expected = torch.stack([torch.nn.functional.linear(activation, w.cuda()) for w in dense], 1).reshape(-1, 64)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
