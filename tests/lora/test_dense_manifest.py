import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.lora import read_adapter, read_runtime_adapter
from freetoken.lora.dense_manifest import read_dense_adapter


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifact(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}")
    for index in range(3):
        (base / f"shard-{index}.safetensors").write_bytes(bytes([index, 1]))
    source = {"kind": "real-dense-checkpoint", "repo": "test/dense", "revision": "a" * 40,
              "metadata": [{"filename": "config.json", "sha256": sha(base / "config.json")}],
              "shards": [{"filename": f"shard-{index}.safetensors", "sha256": sha(base / f"shard-{index}.safetensors")} for index in range(3)]}
    identity = {"family": "qwen3_5_dense", "source": source, "tokenizer_source_hashes": {}}
    target = "model.layers.0.mlp.gate_proj"
    tensors = {f"base_model.model.{target}.lora_{side}.weight": value for side, value in
               [("A", torch.arange(8).reshape(2, 4).float()), ("B", torch.ones(6, 2))]}
    save_file(tensors, adapter / "adapter_model.safetensors")
    peft = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "test/dense",
            "r": 2, "lora_alpha": 4, "target_modules": [target]}
    write(adapter / "adapter_config.json", peft)
    manifest = {"layout_version": "nvfp4-lora-dense-v1", "family": "qwen3_5_dense", "status": "complete",
                "synthetic_vocabulary_only": False, "enable_thinking": False, "base": identity,
                "base_fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "linear_format": "peft-lora", "nonexpert": {"rank": 2, "alpha": 4, "scaling": 2., "rsLoRA": False,
                    "dtype": "float32", "target_modules": [target]},
                "tensor_specifications": {key.removeprefix("base_model.model."): {"file": "adapter_model.safetensors", "key": key,
                    "dtype": "float32", "shape": list(value.shape), "numel": value.numel()} for key, value in tensors.items()},
                "files": {name: sha(adapter / name) for name in ["adapter_config.json", "adapter_model.safetensors"]}}
    write(adapter / "manifest.json", manifest)
    return base, adapter, manifest


def test_dense_load_and_legacy_rejection(artifact):
    base, adapter, _ = artifact
    bundle = read_runtime_adapter(adapter, base)
    assert bundle.parameter_count == 20 and bundle.parameter_bytes == 80 and bundle.experts == {}
    assert bundle.linear["model.layers.0.mlp.gate_proj"].scaling == 2
    with pytest.raises(ValueError, match="Unsupported"):
        read_adapter(adapter, base)


@pytest.mark.parametrize("damage", ["base", "adapter", "fingerprint", "revision", "tokenizer", "target", "missing_b", "expert"])
def test_dense_rejects_incompatible_bytes_and_contract(artifact, damage):
    base, adapter, manifest = artifact
    if damage == "base":
        (base / "shard-1.safetensors").write_bytes(b"different")
    elif damage == "adapter":
        (adapter / "adapter_model.safetensors").write_bytes(b"different")
    elif damage == "fingerprint":
        manifest["base_fingerprint"] = "0" * 64
    elif damage == "revision":
        manifest["base"]["source"]["revision"] = "main"
    elif damage == "tokenizer":
        manifest["base"]["tokenizer_source_hashes"] = {"config.json": "0" * 64}
    elif damage == "target":
        manifest["nonexpert"]["target_modules"] = ["model.layers.0.mlp.up_proj"]
    elif damage == "missing_b":
        name = next(name for name in manifest["tensor_specifications"] if ".lora_B." in name)
        del manifest["tensor_specifications"][name]
    else:
        manifest["tensor_specifications"]["model.layers.0.mlp.experts.gate_proj.lora_A"] = {}
    if damage == "tokenizer":
        manifest["base_fingerprint"] = hashlib.sha256(json.dumps(manifest["base"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(adapter / "manifest.json", manifest)
    with pytest.raises(ValueError):
        read_dense_adapter(adapter, base)
