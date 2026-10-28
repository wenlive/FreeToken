from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from freetoken.layers.linear import LinearReplicated, LinearOProj, LinearRowParallel
from freetoken.lora.bind import bind_adapter, validate_runtime
from freetoken.lora.manifest import AdapterBundle, LoRAPair
from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM


@pytest.mark.parametrize("cls", [LinearReplicated, LinearOProj, LinearRowParallel])
def test_linear_call_paths_apply_adapter(cls):
    torch.manual_seed(5)
    op = cls.__new__(cls)
    x, weight = torch.randn(3, 4).bfloat16(), torch.randn(7, 4).bfloat16()
    op._tp_size = 1
    op.quant_method = NS(apply=lambda _, x: x @ weight.T)
    base = op.forward(x)
    pair = LoRAPair(torch.randn(2, 4), torch.randn(3, 2), 2)
    op._lora_parts = ((1, 4, pair),)
    expected = base.float()
    expected[:, 1:4] += x.float() @ (pair.b @ pair.a * 2).T
    torch.testing.assert_close(op.forward(x), expected.bfloat16(), rtol=0, atol=0)
    op._lora_parts = ((1, 4, LoRAPair(pair.a, torch.zeros_like(pair.b), 2)),)
    assert torch.equal(op.forward(x), base)


def fake_linear(inp, out):
    return NS(in_features=inp, out_features=out)


def test_hf_targets_bind_to_fused_segments_and_are_immutable():
    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    gdn = NS(_split_in_proj=True, conv_dim=6, value_dim=4,
             in_proj_qkvz=fake_linear(4, 10), out_proj=fake_linear(4, 4))
    attn = NS(_qkv_split=[8, 2, 2], qkv_proj=fake_linear(4, 12), o_proj=fake_linear(4, 4))
    def mlp():
        return NS(shared_expert=NS(gate_up_proj=fake_linear(4, 6), down_proj=fake_linear(3, 4)))
    layers = [NS(_is_linear=True, linear_attn=gdn, mlp=mlp()), NS(_is_linear=False, self_attn=attn, mlp=mlp())]
    model.model = NS(layers=NS(op_list=layers))
    targets = {
        "model.layers.0.linear_attn.in_proj_qkv": (gdn.in_proj_qkvz, 0, 6, 4),
        "model.layers.0.linear_attn.in_proj_z": (gdn.in_proj_qkvz, 6, 10, 4),
        "model.layers.1.self_attn.q_proj": (attn.qkv_proj, 0, 8, 4),
        "model.layers.1.self_attn.k_proj": (attn.qkv_proj, 8, 10, 4),
        "model.layers.1.self_attn.v_proj": (attn.qkv_proj, 10, 12, 4),
        "model.layers.0.mlp.shared_expert.up_proj": (layers[0].mlp.shared_expert.gate_up_proj, 3, 6, 4),
        "model.layers.1.mlp.shared_expert.down_proj": (layers[1].mlp.shared_expert.down_proj, 0, 4, 3),
    }
    pairs = {name: LoRAPair(torch.ones(2, inp), torch.full((end - start, 2), index + 1.), 2)
             for index, (name, (_, start, end, inp)) in enumerate(targets.items())}
    bundle = AdapterBundle(Path("."), "adapter-a", {}, pairs, {}, 0, 0)
    report = bind_adapter(model, bundle, "cpu")
    assert report["linear_projections"] == len(targets)
    for name, (op, start, end, _) in targets.items():
        part = next(part for part in op._lora_parts if part[0:2] == (start, end))
        assert torch.equal(part[2].b, pairs[name].b)
    with pytest.raises(ValueError, match="immutable"):
        bind_adapter(model, bundle, "cpu")


@pytest.mark.parametrize("option", ["tp", "dummy", "hybrid", "cpu_layers", "mtp", "multimodal"])
def test_reject_unsupported_runtime(option):
    c = NS(tp_info=NS(size=1), use_dummy_weight=False, moe_strategy="offload", moe_cpu_layers=None,
           mtp_runtime_path=None, moe_q2_resident_experts=0, mm=NS(text_model_only=True))
    if option == "tp": c.tp_info.size = 2
    elif option == "dummy": c.use_dummy_weight = True
    elif option == "hybrid": c.moe_strategy = "hybrid"
    elif option == "cpu_layers": c.moe_cpu_layers = "3"
    elif option == "mtp": c.mtp_runtime_path = "draft"
    else: c.mm.text_model_only = False
    with pytest.raises(ValueError, match="LoRA currently requires"):
        validate_runtime(c)
