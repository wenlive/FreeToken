"""Recheck the actual official file's tokenizer, storage and source conventions."""
import argparse
import json
from pathlib import Path
import re

import gguf
import numpy as np
import torch
from transformers import AutoTokenizer

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.layers import rotary
from freetoken.models import create_model
from freetoken.models.clover_moe import gguf as adapter
from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.models.gguf.tokenizer import load_gguf_tokenizer
from freetoken.models.loader import ShardReader
from freetoken.models.qwen3_5_moe.weight import _is_gemma_norm
from freetoken.utils.torch_utils import torch_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
model = "C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf"
source = "C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff"
set_tp_info(0, 1)
engine = EngineConfig(model_path=model, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy="offload")
config = engine.model_config
object.__setattr__(config, "moe_strategy", "offload")
rotary.set_rope_device(torch.device("cpu"))
rotary.get_rope.cache_clear()
with torch.device("meta"), torch_dtype(torch.bfloat16):
    expected = create_model(config).state_dict()
observed, packed_bytes, dense_bytes = set(), 0, 0
for name, tensor in adapter.iter_gguf_weights(model, torch.device("cpu"), include_moe_experts=False, include_non_moe=True):
    assert name in expected and name not in observed, name
    assert tensor.shape == expected[name].shape, (name, tensor.shape, expected[name].shape)
    if expected[name].dtype == torch.uint8:
        assert tensor.dtype == torch.uint8
        packed_bytes += tensor.numel()
    else:
        assert tensor.dtype == torch.bfloat16 and torch.isfinite(tensor).all(), name
        dense_bytes += tensor.numel() * tensor.element_size()
    observed.add(name)
assert observed == set(expected), sorted(set(expected) - observed)
assert all(pair == (8, 8) for pair in config.expert_source["types"])
print(json.dumps(dict(dense_buffers=len(observed), packed_dense_bytes=packed_bytes)), flush=True)

reader = ShardReader(source, torch.device("cpu"))
exact, norm_rounding = [], []
try:
    for tensor in iter_gguf_tensors(model):
        if tensor.ggml_type not in (0, 30) or "_exps" in tensor.name:
            continue
        if tensor.name in ("token_embd.weight", "output.weight"):
            continue
        if tensor.name == "output_norm.weight":
            name = "model.norm.weight"
        else:
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
            name = f"model.layers.{int(match[1])}.{adapter._LAYER[match[2]]}"
        value = gguf.dequantize(tensor._raw, gguf.GGMLQuantizationType(tensor.ggml_type))
        restored = adapter._restore_dense(name, torch.from_numpy(np.array(value, copy=True)).reshape(tensor.shape), config)
        original = reader.get_tensor(name).bfloat16()
        if _is_gemma_norm(name):
            assert torch.equal(restored.float() + 1., original.float() + 1.), name
            if not torch.equal(restored, original):
                norm_rounding.append(name)
        else:
            assert torch.equal(restored, original), name
        exact.append(name)
finally:
    reader.close()
native = load_gguf_tokenizer(model)
gold = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
assert all(native.convert_tokens_to_ids(token) == index for token, index in gold.get_vocab().items())
probes = ["Count from 1 to 10.", "Cafe\u0301 café 中文 👩🏽‍💻", "<think>\n</think>\n<|im_end|>", "1234 0.125\t\n"]
for text in probes:
    assert native.encode(text, add_special_tokens=False) == gold.encode(text, add_special_tokens=False)
    for thinking in (True, False):
        messages = [dict(role="user", content=text)]
        assert native.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking, return_dict=False) == gold.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking, return_dict=False)
result = dict(state="complete", model=model, source=source, dense_buffers=len(observed),
    packed_dense_bytes=packed_bytes, other_dense_bytes=dense_bytes,
    exact_source_parameters=len(exact), exact_names=exact, norm_delta_rounding=norm_rounding,
    tokenizer_ids_checked=len(gold.get_vocab()), tokenizer_probes=len(probes),
    cuda_initialized=torch.cuda.is_initialized(), expert_types=config.expert_source["types"],
    conventions="inverse even/odd V-head order; raw A_log; folded +1 norms; plain ssm_norm; separate packed gate/up/down")
(args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k not in ("exact_names", "expert_types")}), flush=True)
