"""Independent per-route reference, including cache permutation and BF16 boundaries."""

import pytest
import torch
import torch.nn.functional as F

from freetoken.lora.manifest import LoRAPair
from freetoken.lora.ops import gate_up_residual, pack_experts
from freetoken.moe.fused import moe_align_block_size
from freetoken.moe.fused_nvfp4 import (
    _decode_gemm_marlin, _prefill_config, _prefill_gemm,
    fused_experts_decode_nvfp4_marlin, fused_experts_nvfp4,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def dequant(packed, scales, glob):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack([packed & 15, packed >> 4], -1).flatten(-2).long()
    return lut[codes] * scales.float().repeat_interleave(16, -1) * glob.float().unsqueeze(-1)


def assert_bf16_pipeline_close(actual, expected):
    # FP32 reductions and fast SiLU can land on opposite sides of a BF16 tie.
    # Route cancellation makes pointwise relative error ill-conditioned near zero.
    delta = actual.float() - expected.float()
    assert delta.norm() <= .001 * expected.float().norm()
    assert delta.abs().max() <= .01 * expected.float().abs().max()


def assert_fp32_accumulator_close(actual, expected):
    delta = actual.cpu() - expected
    assert delta.norm() <= 2e-6 * expected.norm()
    assert delta.abs().max() <= 1e-5 * expected.abs().max() + 1e-6


@pytest.mark.parametrize("tokens,h,i", [(1, 64, 32), (17, 64, 32), (65, 64, 32),
                                        (1, 2048, 512), (17, 2048, 512)])
def test_nvfp4_lora_prefill_decode_graph_and_zero(tokens, h, i):
    torch.manual_seed(81)
    torch.backends.cuda.matmul.allow_tf32 = False
    e, k = 5, 2
    gu = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8)
    dn = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)
    gs = torch.ones(e, 2 * i, h // 16).to(torch.float8_e4m3fn)
    ds = torch.ones(e, h, i // 16).to(torch.float8_e4m3fn)
    gg = torch.full((e, 2 * i), .125, dtype=torch.float16)
    gg[:, i:] *= .5
    dg = torch.full((e, h), .0625, dtype=torch.float16)
    wg, wd = dequant(gu, gs, gg), dequant(dn, ds, dg)
    pairs = {}
    for name in ("gate", "up", "down"):
        inp, out = (i, h) if name == "down" else (h, i)
        pairs[name] = LoRAPair(torch.randn(e, 2, inp) * .1, torch.randn(e, out, 2) * .1, 2)
    ids = torch.randint(0, e, (tokens, k), dtype=torch.int32)
    weights = torch.softmax(torch.randn(tokens, k), -1)
    x = torch.randn(tokens, h).bfloat16()
    expected = torch.zeros(tokens, h)
    first_reference = torch.empty(tokens, k, 2 * i)
    for t in range(tokens):
        routes = []
        for j in range(k):
            expert = int(ids[t, j])
            projections = []
            for index, name in enumerate(("gate", "up")):
                p = pairs[name]
                base = wg[expert, index * i:(index + 1) * i] @ x[t].float()
                delta = p.b[expert] @ (p.a[expert] @ x[t].float()) * p.scaling
                first_reference[t, j, index * i:(index + 1) * i] = base + delta
                projections.append((base + delta).bfloat16().float())
            act = (F.silu(projections[0]) * projections[1]).bfloat16().float()
            p = pairs["down"]
            out = wd[expert] @ act + p.b[expert] @ (p.a[expert] @ act) * p.scaling
            routes.append((out * weights[t, j]).bfloat16().float())
        expected[t] = torch.stack(routes).sum(0)
    expected = expected.bfloat16()
    gpu_pairs = pack_experts(pairs, "cuda")
    zero_pairs = {name: LoRAPair(p.a, torch.zeros_like(p.b), p.scaling) for name, p in gpu_pairs.items()}
    gx, gi, weights = x.cuda(), ids.cuda(), weights.cuda()
    banks = [t.cuda() for t in (gu, gs, gg, dn, ds, dg)]
    prefill = fused_experts_nvfp4(gx, *banks, weights, gi, e, lora=gpu_pairs, logical_ids=gi)
    assert_bf16_pipeline_close(prefill.cpu(), expected)
    cfg = _prefill_config(tokens)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(gi, cfg["BLOCK_SIZE_M"], e)
    delta = gate_up_residual(gx, gi, gpu_pairs)
    first_accumulator = torch.empty(tokens, k, 2 * i, device="cuda", dtype=torch.float32)
    _prefill_gemm(gx, *banks[:3], first_accumulator, weights.reshape(-1), sorted_ids, expert_ids,
                  ntpp, tokens * k, k, False, cfg, delta)
    assert_fp32_accumulator_close(first_accumulator, first_reference)
    base = fused_experts_nvfp4(gx, *banks, weights, gi, e)
    zero = fused_experts_nvfp4(gx, *banks, weights, gi, e, lora=zero_pairs, logical_ids=gi)
    assert torch.equal(base, zero)
    assert not torch.equal(base, prefill)
    # Slot bank order differs from logical ids; this catches using slot ids for LoRA.
    order = torch.tensor([3, 0, 4, 1, 2], device="cuda")
    inverse = torch.argsort(order)
    slots = inverse[gi.long()].int()
    permuted = [t.index_select(0, order) for t in banks]
    _decode_gemm_marlin(gx, *permuted[:3], first_accumulator, weights, slots, False, False, delta)
    assert_fp32_accumulator_close(first_accumulator, first_reference)
    def decode():
        return fused_experts_decode_nvfp4_marlin(gx, *permuted, weights, slots,
                                                lora=gpu_pairs, logical_ids=gi)
    out = decode()
    assert_bf16_pipeline_close(out.cpu(), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = decode()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, out)
