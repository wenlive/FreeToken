"""Independent byte/LUT references for the offline Clover checkpoint format."""
import json

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from freetoken.checkpoint.clover_nvfp4 import convert, dequantize, encode_e2m1, pack_e2m1, quantize


def test_e2m1_codes_nibbles_and_signs():
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    codes = encode_e2m1(values)
    assert codes.tolist() == list(range(16))
    assert pack_e2m1(codes).tolist() == [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
    actual = dequantize(pack_e2m1(codes).view(1, -1), torch.ones(1, 1).to(torch.float8_e4m3fn), torch.tensor(1.))
    torch.testing.assert_close(actual.flatten(), values, rtol=0, atol=0)


def test_e2m1_ties_to_even_and_saturation():
    values = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 100.])
    assert encode_e2m1(values).tolist() == [0, 2, 2, 4, 4, 6, 6, 7]
    assert encode_e2m1(-values).tolist() == [8, 10, 10, 12, 12, 14, 14, 15]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantization_matches_independent_byte_decoder(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    original = torch.randn(64, 128, generator=torch.Generator().manual_seed(8)).to(torch.bfloat16).to(device)
    packed, scales, glob = quantize(original)
    assert float(glob) == float(glob.half())
    raw, s, g = packed.cpu().tolist(), scales.float().cpu().tolist(), float(glob)
    signed = [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]
    reference = [[signed[(row[k // 2] >> (4 * (k % 2))) & 15] * s[r][k // 16] * g for k in range(128)] for r, row in enumerate(raw)]
    torch.testing.assert_close(dequantize(packed, scales, glob).cpu(), torch.tensor(reference), rtol=0, atol=0)
    relative = (dequantize(packed, scales, glob) - original.float()).norm() / original.float().norm()
    assert 0.06 < float(relative) < 0.13


def test_zero_and_invalid_weights():
    packed, scales, glob = quantize(torch.zeros(2, 32))
    assert dequantize(packed, scales, glob).count_nonzero() == 0
    with pytest.raises(ValueError, match="divisible"):
        quantize(torch.ones(2, 17))
    with pytest.raises(ValueError, match="non-finite"):
        quantize(torch.full((2, 16), float("nan")))


def test_streaming_converter_preserves_nonexperts_and_auxiliary_files(tmp_path):
    source, output = tmp_path / "source", tmp_path / "quantized"
    source.mkdir()
    config = dict(model_type="clover_moe", num_hidden_layers=1, num_experts=2, hidden_size=32, moe_intermediate_size=16)
    (source / "config.json").write_text(json.dumps(config))
    dense = torch.arange(32).to(torch.bfloat16)
    data = {"model.norm.weight": dense, "model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 32, 32).bfloat16(),
            "model.layers.0.mlp.experts.down_proj": torch.randn(2, 32, 16).bfloat16()}
    save_file(data, source / "model.safetensors")
    aux = {"model.layers.0.linear_attn.conv_right_weight": torch.randn(32, 1, 3).bfloat16()}
    save_file(aux, source / "bd_gdn_conv_right_weight.safetensors")
    names = {**{n: "model.safetensors" for n in data}, **{n: "bd_gdn_conv_right_weight.safetensors" for n in aux}}
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    result = convert(source, output, device="cpu", sample_experts=2)
    assert result["state"] == "complete" and result["tensor_count"] == 20
    assert (output / "bd_gdn_conv_right_weight.safetensors").read_bytes() == (source / "bd_gdn_conv_right_weight.safetensors").read_bytes()
    index = json.loads((output / "model.safetensors.index.json").read_text())
    with safe_open(output / index["weight_map"]["model.norm.weight"], framework="pt") as handle:
        torch.testing.assert_close(handle.get_tensor("model.norm.weight"), dense, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        convert(source, output, device="cpu")
