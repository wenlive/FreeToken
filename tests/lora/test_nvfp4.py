"""Independent per-route reference, including cache permutation and BF16 boundaries."""

import pytest
import torch
import torch.nn.functional as F

from freetoken.lora.manifest import LoRAPair
from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin, fused_experts_nvfp4


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def dequant(packed, scales, glob):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack([packed & 15, packed >> 4], -1).flatten(-2).long()
    return lut[codes] * scales.float().repeat_interleave(16, -1) * glob.float().unsqueeze(-1)


@pytest.mark.parametrize("tokens", [1, 17, 65])
def test_nvfp4_lora_prefill_decode_graph_and_zero(tokens):
    torch.manual_seed(81)
    torch.backends.cuda.matmul.allow_tf32 = False
    e, h, i, k = 5, 64, 32, 2
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
    for t in range(tokens):
        routes = []
        for j in range(k):
            expert = int(ids[t, j])
            projections = []
            for index, name in enumerate(("gate", "up")):
                p = pairs[name]
                base = wg[expert, index * i:(index + 1) * i] @ x[t].float()
                delta = p.b[expert] @ (p.a[expert] @ x[t].float()) * p.scaling
                projections.append((base + delta).bfloat16().float())
            act = (F.silu(projections[0]) * projections[1]).bfloat16().float()
            p = pairs["down"]
            out = wd[expert] @ act + p.b[expert] @ (p.a[expert] @ act) * p.scaling
            routes.append((out * weights[t, j]).bfloat16().float())
        expected[t] = torch.stack(routes).sum(0)
    expected = expected.bfloat16()
    gpu_pairs = {name: p.to("cuda") for name, p in pairs.items()}
    zero_pairs = {name: LoRAPair(p.a, torch.zeros_like(p.b), p.scaling) for name, p in gpu_pairs.items()}
    gx, gi, weights = x.cuda(), ids.cuda(), weights.cuda()
    banks = [t.cuda() for t in (gu, gs, gg, dn, ds, dg)]
    prefill = fused_experts_nvfp4(gx, *banks, weights, gi, e, lora=gpu_pairs, logical_ids=gi)
    torch.testing.assert_close(prefill.cpu(), expected, rtol=.02, atol=.006)
    base = fused_experts_nvfp4(gx, *banks, weights, gi, e)
    zero = fused_experts_nvfp4(gx, *banks, weights, gi, e, lora=zero_pairs, logical_ids=gi)
    assert torch.equal(base, zero)
    assert not torch.equal(base, prefill)
    # Slot bank order differs from logical ids; this catches using slot ids for LoRA.
    order = torch.tensor([3, 0, 4, 1, 2], device="cuda")
    inverse = torch.argsort(order)
    slots = inverse[gi.long()].int()
    permuted = [t.index_select(0, order) for t in banks]
    def decode():
        return fused_experts_decode_nvfp4_marlin(gx, *permuted, weights, slots,
                                                lora=gpu_pairs, logical_ids=gi)
    out = decode()
    torch.testing.assert_close(out.cpu(), expected, rtol=.02, atol=.006)
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
