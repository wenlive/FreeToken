import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.lora import read_adapter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def artifact(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}")
    (base / "weights.bin").write_bytes(b"packed checkpoint")
    source = {"kind": "real-nvidia-checkpoint", "metadata": [{"filename": "config.json", "sha256": sha(base / "config.json")}],
              "shards": [{"filename": "weights.bin", "sha256": sha(base / "weights.bin")}]}
    identity = {"source": source}
    target = "model.layers.0.self_attn.q_proj"
    tensors = {f"base_model.model.{target}.lora_{side}.weight": tensor for side, tensor in
               [("A", torch.arange(8).reshape(2, 4).float()), ("B", torch.ones(6, 2))]}
    save_file(tensors, adapter / "adapter_model.safetensors")
    save_file({}, adapter / "expert_lora.safetensors")
    config = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": "test"}
    write_json(adapter / "adapter_config.json", config)
    manifest = {"layout_version": "nvfp4-lora-v1", "status": "complete", "synthetic_vocabulary_only": False,
                "enable_thinking": False, "base": identity,
                "base_fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "nonexpert": {"rank": 2, "alpha": 4, "scaling": 2, "rsLoRA": False, "dtype": "float32", "target_modules": "test"},
                "experts": {}, "expert_layout": "{gate,up,down}_proj.lora_A[E,r,in] and lora_B[E,out,r]",
                "linear_format": "peft-lora", "tensor_specifications": {
                    key.removeprefix("base_model.model."): {"file": "adapter_model.safetensors", "key": key,
                        "shape": list(t.shape), "dtype": "float32", "numel": t.numel()} for key, t in tensors.items()},
                "files": {name: sha(adapter / name) for name in
                          ["adapter_config.json", "adapter_model.safetensors", "expert_lora.safetensors"]}}
    write_json(adapter / "manifest.json", manifest)
    return base, adapter, manifest


def test_load_canonical_pairs_and_scaling(artifact):
    base, adapter, _ = artifact
    bundle = read_adapter(adapter, base)
    assert bundle.parameter_count == 20 and bundle.parameter_bytes == 80
    assert len(bundle.linear) == 1 and bundle.experts == {}
    pair = bundle.linear["model.layers.0.self_attn.q_proj"]
    assert pair.rank == 2 and pair.scaling == 2
    assert torch.equal(pair.a, torch.arange(8).reshape(2, 4).float())


def test_load_ignores_legacy_hashes_and_does_not_scan_base_shards(artifact):
    """Real tensor edits remain readable without refreshing obsolete SHA metadata."""
    from safetensors.torch import load_file

    base, adapter, manifest = artifact
    (base / "weights.bin").unlink()
    manifest["base_fingerprint"] = "unused legacy fingerprint"
    manifest["files"] = {name: "unused legacy checksum" for name in manifest["files"]}
    tensors = load_file(adapter / "adapter_model.safetensors")
    key = next(key for key in tensors if "lora_B" in key)
    tensors[key].fill_(7.0)
    save_file(tensors, adapter / "adapter_model.safetensors")
    write_json(adapter / "manifest.json", manifest)
    bundle = read_adapter(adapter, base)
    assert torch.equal(bundle.linear["model.layers.0.self_attn.q_proj"].b, torch.full((6, 2), 7.0))


@pytest.mark.parametrize("damage", ["missing_file", "scaling", "canonical", "missing_pair", "nan"])
def test_reject_invalid_artifact(artifact, damage):
    base, adapter, m = artifact
    if damage == "missing_file":
        (adapter / "adapter_model.safetensors").unlink()
    elif damage == "scaling":
        m["nonexpert"]["scaling"] = 1
    elif damage == "canonical":
        next(iter(m["tensor_specifications"].values()))["key"] = "wrong"
    else:
        from safetensors.torch import load_file
        tensors = load_file(adapter / "adapter_model.safetensors")
        if damage == "missing_pair":
            key = next(k for k in tensors if "lora_B" in k)
            tensors.pop(key)
            m["tensor_specifications"].pop(key.removeprefix("base_model.model."))
        else:
            next(iter(tensors.values())).flatten()[0] = float("nan")
        save_file(tensors, adapter / "adapter_model.safetensors")
        m["files"]["adapter_model.safetensors"] = sha(adapter / "adapter_model.safetensors")
    write_json(adapter / "manifest.json", m)
    with pytest.raises(ValueError):
        read_adapter(adapter, base)
