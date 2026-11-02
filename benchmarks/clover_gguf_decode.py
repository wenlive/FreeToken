"""Independent eager-Torch GGUF reference decoder (no serving CUDA kernels).

Format reference: ggml-org/llama.cpp@4838ffb0c8ade92f34d9157deea38d92758ab824,
gguf-py/gguf/quants.py, Q4_K and IQ4_XS (MIT).
Qualified bitwise against installed gguf-py0.19.0's NumPy decoder.
Copyright (c) 2023-2026 The ggml authors; see LICENSE.clover-gguf-reference.
"""

import numpy as np
import torch


def decode_expert(raw, kind, *, device):
    if kind not in (12, 23):
        raise ValueError(f"Unsupported independent GGUF expert type {kind}")
    size = 144 if kind == 12 else 136
    rows = raw.shape[0]
    packed = torch.from_numpy(np.array(raw, copy=True)).to(device).reshape(-1, size)
    scale = packed[:, :2].contiguous().view(torch.float16).float()
    if kind == 23:
        high = packed[:, 2:4].contiguous().view(torch.int16).int()
        low = packed[:, 4:8]
        low = torch.stack((low & 15, low >> 4), dim=-1).reshape(-1, 8)
        high = (high >> (2 * torch.arange(8, device=device))) & 3
        scales = (low.int() | (high << 4)) - 32
        codes = packed[:, 8:].reshape(-1, 8, 16)
        codes = torch.cat((codes & 15, codes >> 4), dim=-1)
        values = torch.tensor(
            [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
            device=device,
            dtype=torch.float32,
        )
        result = (scale * scales.float()).unsqueeze(-1) * values[codes.long()]
    else:
        minimum = packed[:, 2:4].contiguous().view(torch.float16).float()
        metadata = packed[:, 4:16].reshape(-1, 3, 4)
        lo, mid, hi = metadata.unbind(1)
        scales = torch.cat((lo & 63, (hi & 15) | ((lo >> 2) & 48)), dim=-1)
        mins = torch.cat((mid & 63, (hi >> 4) | ((mid >> 2) & 48)), dim=-1)
        codes = packed[:, 16:].reshape(-1, 4, 32)
        codes = torch.stack((codes & 15, codes >> 4), dim=2).reshape(-1, 8, 32)
        result = (scale * scales.float()).unsqueeze(-1) * codes.float() - (
            minimum * mins.float()
        ).unsqueeze(-1)
    return result.reshape(rows, -1)
