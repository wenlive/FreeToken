"""Inference adapters for the versioned NVFP4 training artifact format."""

from .manifest import AdapterBundle, LoRAPair, read_adapter


def read_runtime_adapter(directory, checkpoint):
    from pathlib import Path
    from .manifest import _read

    if _read(Path(directory) / "manifest.json").get("layout_version") == "nvfp4-lora-dense-v1":
        from .dense_manifest import read_dense_adapter

        return read_dense_adapter(directory, checkpoint)
    return read_adapter(directory, checkpoint)

__all__ = ["AdapterBundle", "LoRAPair", "read_adapter"]
