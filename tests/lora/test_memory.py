"""Compare header admission with the tensors actually packed by inference."""
import torch
import pytest
from safetensors.torch import save_file

from freetoken.lora.manifest import LoRAPair
from freetoken.lora.ops import pack_experts, pack_linear
from freetoken.lora_memory import adapter_resident_bytes


def test_header_estimate_matches_fused_and_expert_allocations(tmp_path):
    config = dict(num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                  linear_num_key_heads=1, linear_key_head_dim=2,
                  linear_num_value_heads=1, linear_value_head_dim=4,
                  shared_expert_intermediate_size=5)
    linear, expert, actual = {}, {}, 0
    groups = [('self_attn', [('q_proj', 16), ('k_proj', 4), ('v_proj', 4)]),
              ('linear_attn', [('in_proj_qkv', 8), ('in_proj_z', 4)]),
              ('mlp.shared_expert', [('gate_proj', 5), ('up_proj', 5)])]
    for namespace, segments in groups:
        parts, offset = [], 0
        for index, (name, width) in enumerate(segments):
            rank = index + 1
            pair = LoRAPair(torch.ones(rank, 7), torch.ones(width, rank), 2.0)
            prefix = f'base_model.model.model.layers.0.{namespace}.{name}'
            linear[prefix+'.lora_A.weight'] = pair.a
            linear[prefix+'.lora_B.weight'] = pair.b
            parts.append((offset, offset+width, pair))
            offset += width
        packed = pack_linear(parts, 7, offset, 'cpu')
        actual += sum(t.numel()*t.element_size() for t in (packed.a, packed.b))
    pairs = {}
    for name, inp, out in [('gate', 7, 5), ('up', 7, 5), ('down', 5, 7)]:
        pair = LoRAPair(torch.ones(3, 2, inp), torch.ones(3, out, 2), 2.0)
        pairs[name] = pair
        prefix = f'model.layers.0.mlp.experts.{name}_proj'
        expert[prefix+'.lora_A'], expert[prefix+'.lora_B'] = pair.a, pair.b
    actual += sum(t.numel()*t.element_size() for p in pack_experts(pairs, 'cpu').values() for t in (p.a, p.b))
    save_file(linear, tmp_path/'adapter_model.safetensors')
    save_file(expert, tmp_path/'expert_lora.safetensors')
    assert adapter_resident_bytes(tmp_path, {'text_config': config}) == actual


def test_admission_rejects_unqualified_factor_dtype(tmp_path):
    save_file({'model.layers.0.self_attn.o_proj.lora_A.weight': torch.ones(2, 7).bfloat16()},
              tmp_path/'adapter_model.safetensors')
    with pytest.raises(ValueError, match='requires F32'):
        adapter_resident_bytes(tmp_path, {})
