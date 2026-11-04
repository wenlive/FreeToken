"""Independent eager-Torch GGUF reference decoder (no serving CUDA kernels).

Format reference: ggml-org/llama.cpp@4838ffb0c8ade92f34d9157deea38d92758ab824,
gguf-py/gguf/quants.py, Q4_K and IQ4_XS (MIT).
Additional formats: Maphist0/llama.cpp-pub@cb09d4355d8bf968bb907c5de03cf42d8aa073ce,
gguf-py/gguf/quants.py, Q8_0/Q4_0/Q6_K (MIT); ported to eager Torch below.
Qualified bitwise against installed gguf-py0.19.0's NumPy decoder.
Copyright (c) 2023-2026 The ggml authors; see LICENSE.clover-gguf-reference.
"""

import numpy as np
import torch


def decode_expert(raw, kind, *, device):
    if kind not in (2, 8, 12, 14, 23):
        raise ValueError(f"Unsupported independent GGUF expert type {kind}")
    size = {2: 18, 8: 34, 12: 144, 14: 210, 23: 136}[kind]
    rows = raw.shape[0]
    packed = torch.from_numpy(np.array(raw, copy=True)).to(device).reshape(-1, size)
    if kind == 14:
        scale = packed[:, -2:].contiguous().view(torch.float16).float()
        scales = packed[:, 192:208].contiguous().view(torch.int8).float()
        low = packed[:, :128].reshape(-1, 2, 1, 64)
        low = ((low >> torch.tensor([0, 4], device=device).reshape(1, 1, 2, 1)) & 15).reshape(-1, 8, 32)
        high = packed[:, 128:192].reshape(-1, 2, 1, 32)
        high = ((high >> torch.tensor([0, 2, 4, 6], device=device).reshape(1, 1, 4, 1)) & 3).reshape(-1, 8, 32)
        codes = ((low | (high << 4)).int() - 32).reshape(-1, 16, 16).float()
        return ((scale * scales).unsqueeze(-1) * codes).reshape(rows, -1)
    scale = packed[:, :2].contiguous().view(torch.float16).float()
    if kind == 8:
        return (scale * packed[:, 2:].contiguous().view(torch.int8).float()).reshape(rows, -1)
    if kind == 2:
        codes = torch.cat((packed[:, 2:] & 15, packed[:, 2:] >> 4), -1).float() - 8
        return (scale * codes).reshape(rows, -1)
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
