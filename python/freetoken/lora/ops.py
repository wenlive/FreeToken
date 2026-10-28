"""FP32 residuals over the inference backend's unmodified packed base weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .manifest import LoRAPair


@dataclass(frozen=True)
class PackedLinear:
    a: torch.Tensor
    b: torch.Tensor
    segments: tuple[tuple[int, int, int], ...]


def pack_linear(parts, input_size: int, output_size: int, device) -> PackedLinear:
    """Fold scaling and compile independent output segments into two matrices."""
    total_rank = sum(pair.rank for _, _, pair in parts)
    a = torch.cat([pair.a for _, _, pair in parts], dim=0)
    b = torch.zeros(output_size, total_rank, dtype=torch.float32)
    offset = 0
    for start, end, pair in parts:
        b[start:end, offset:offset + pair.rank] = pair.b * pair.scaling
        offset += pair.rank
    return PackedLinear(a.to(device).contiguous(), b.to(device).contiguous(),
                        tuple((start, end, pair.rank) for start, end, pair in parts))


def linear_residual(x: torch.Tensor, y: torch.Tensor, packed: PackedLinear) -> torch.Tensor:
    if x.is_cuda and x.numel() // x.shape[-1] <= 8:
        from .kernels import residual

        return residual(x.reshape(-1, x.shape[-1]), packed.a, packed.b,
                        base=y.reshape(-1, y.shape[-1])).reshape(y.shape)
    low = F.linear(x.float(), packed.a)
    result = y.reshape(-1, y.shape[-1]).float()
    result.addmm_(low.reshape(-1, low.shape[-1]), packed.b.T)
    return result.reshape(y.shape).to(y.dtype)


def pack_experts(pairs, device):
    """Compile gate/up together; retain only the packed FP32 runtime factors."""
    gate, up, down = (pairs[name] for name in ("gate", "up", "down"))
    e, out, rank = gate.b.shape
    a = torch.cat([gate.a, up.a], dim=1)
    b = torch.zeros(e, out + up.b.shape[1], rank + up.rank, dtype=torch.float32)
    b[:, :out, :rank] = gate.b * gate.scaling
    b[:, out:, rank:] = up.b * up.scaling
    return {"gate_up": LoRAPair(a.to(device).contiguous(), b.to(device).contiguous(), 1.0),
            "down": LoRAPair(down.a.to(device).contiguous(),
                              (down.b * down.scaling).to(device).contiguous(), 1.0)}


def expert_residual(x: torch.Tensor, ids: torch.Tensor, pair: LoRAPair, *, routed: bool) -> torch.Tensor:
    """Return [routes, out] deltas indexed by logical experts, never cache slots.

    Bound the gathered weights during prefill; loop lengths depend only on shapes,
    so the same code is safe during decode CUDA graph capture.
    """
    if x.is_cuda:
        from .kernels import residual

        return residual(x, pair.a, pair.b, ids=ids, routed=routed, scaling=pair.scaling)
    flat = ids.reshape(-1).long()
    result = torch.empty((flat.numel(), pair.b.shape[1]), dtype=torch.float32, device=x.device)
    top_k = ids.shape[1]
    # Include gather, FP32 input/output and low-rank scratch in the 64 MiB bound.
    per_route = 4 * (pair.a.shape[1] * pair.a.shape[2] + pair.b.shape[1] * pair.b.shape[2]
                     + pair.a.shape[2] + pair.b.shape[1] + pair.rank)
    chunk = max(1, (64 * 1024**2 // per_route) // 256 * 256)
    for start in range(0, flat.numel(), chunk):
        end = min(start + chunk, flat.numel())
        logical = flat[start:end]
        rows = torch.arange(start, end, device=x.device)
        inputs = x.index_select(0, rows if routed else rows // top_k).float()
        a = pair.a.index_select(0, logical)
        b = pair.b.index_select(0, logical)
        low = torch.bmm(a, inputs.unsqueeze(-1))
        result[start:end].copy_(torch.bmm(b, low).squeeze(-1) * pair.scaling)
    return result


def gate_up_residual(x, ids, pairs):
    return expert_residual(x, ids, pairs["gate_up"], routed=False)
