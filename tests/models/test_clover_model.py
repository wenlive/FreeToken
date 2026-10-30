"""Native-key Clover loading against an independently constructed tiny checkpoint."""
import json

import pytest
from safetensors.torch import save_file
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import set_quant_config
from freetoken.models.clover_moe import parse_config, iter_weights, nvfp4_expert_spec
from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces
from freetoken.models.register import checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config


@pytest.fixture
def checkpoint(tmp_path):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = dict(model_type="clover_moe", architectures=["CloverMoeModelLM"],
                  hidden_size=128, vocab_size=256, num_hidden_layers=2, intermediate_size=128,
                  num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=128,
                  head_dim=64, num_attention_heads=4, num_key_value_heads=2,
                  linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
                  linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"],
                  hidden_act="silu", rms_norm_eps=1e-6, max_position_embeddings=1024,
                  rope_parameters=dict(rope_type="default", rope_theta=10000., partial_rotary_factor=.25),
                  quantization_config=dict(quant_method="modelopt", quant_algo="MIXED_PRECISION", quantized_layers={
                      f"model.layers.{i}.mlp.experts": dict(quant_algo="W4A16_NVFP4", group_size=16) for i in range(2)}))
    raw = {}
    def add(name, *shape):
        raw[name] = torch.randn(*shape).bfloat16()
    add("model.embed_tokens.weight", 256, 128)
    add("lm_head.weight", 256, 128)
    add("model.norm.weight", 128)
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            add(f"{prefix}.{norm}.weight", 128)
        add(f"{prefix}.mlp.gate.weight", 4, 128)
        add(f"{prefix}.mlp.shared_expert_gate.weight", 1, 128)
        for proj in ("gate", "up", "down"):
            add(f"{prefix}.mlp.shared_expert.{proj}_proj.weight", 128, 128)
            for expert in range(4):
                pre = f"{prefix}.mlp.experts.{expert}.{proj}_proj"
                raw[f"{pre}.weight"] = torch.randint(0, 256, (128, 64), dtype=torch.uint8)
                raw[f"{pre}.weight_scale"] = torch.ones(128, 8).to(torch.float8_e4m3fn)
                raw[f"{pre}.weight_scale_2"] = torch.tensor(.125)
    pre = "model.layers.0.linear_attn"
    for proj, size in (("qkv", 512), ("z", 256), ("b", 4), ("a", 4)):
        add(f"{pre}.in_proj_{proj}.weight", size, 128)
    add(f"{pre}.out_proj.weight", 128, 256)
    add(f"{pre}.conv1d.weight", 512, 1, 4)
    add(f"{pre}.conv_right_weight", 512, 1, 3)
    for name in ("A_log", "dt_bias", "bd_softmax_logit_scale", "bd_softmax_history_bias", "bd_softmax_history_c"):
        add(f"{pre}.{name}", 4)
    add(f"{pre}.norm.weight", 64)
    pre = "model.layers.1.self_attn"
    for proj, rows, cols in (("q", 512, 128), ("k", 128, 128), ("v", 128, 128), ("o", 128, 256)):
        add(f"{pre}.{proj}_proj.weight", rows, cols)
    add(f"{pre}.q_norm.weight", 64)
    add(f"{pre}.k_norm.weight", 64)
    names = sorted(raw)
    weight_map = {}
    for shard in range(2):
        filename = f"model-{shard}.safetensors"
        tensors = {name: raw[name] for name in names[shard::2]}
        save_file(tensors, tmp_path / filename)
        weight_map.update({name: filename for name in tensors})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=weight_map)))
    hf = cached_load_hf_config(str(tmp_path))
    set_quant_config(checkpoint_quant_config(str(tmp_path), hf, get_model_spec("CloverMoeModelLM")))
    return str(tmp_path), hf, raw


def test_clover_expert_probe_uses_native_prefix(checkpoint):
    _, hf, _ = checkpoint
    config = parse_config(hf)
    assert config.expert_quant == "nvfp4" and config.model_type == "clover_moe"
    assert config.linear_attention_group().layer_ids == (0,)


def test_clover_dense_buffers_and_auxiliaries_match_checkpoint(checkpoint):
    path, hf, raw = checkpoint
    from freetoken.engine.config import EngineConfig
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    config = EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy="offload")
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", "gpu")
    previous = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            buffers = create_model(config.model_config).state_dict()
    finally:
        rotary.set_rope_device(previous)
        rotary.get_rope.cache_clear()
    actual = dict(iter_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    assert set(actual) == set(buffers)
    for name, tensor in actual.items():
        assert tensor.shape == buffers[name].shape
    torch.testing.assert_close(actual["model.norm.weight"], raw["model.norm.weight"] + 1., rtol=0, atol=0)
    key = "model.layers.0.linear_attn.conv_right_weight"
    torch.testing.assert_close(actual[key], raw[key], rtol=0, atol=0)


def test_clover_expert_bank_pieces_keep_layer_and_expert_identity(checkpoint):
    path, hf, _ = checkpoint
    config = parse_config(hf)
    spec = nvfp4_expert_spec(path, config)
    pieces = list(iter_nvfp4_expert_pieces(path, config, spec))
    assert len(pieces) == 8
    assert {(layer, begin) for layer, begin, end, tensors in pieces} == {(l, e) for l in range(2) for e in range(4)}
