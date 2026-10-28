"""FP32 residuals over the inference backend's unmodified packed base weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .manifest import LoRAPair


def linear_residual(x: torch.Tensor, y: torch.Tensor, parts) -> torch.Tensor:
    result = y.float()
    for start, end, pair in parts:
        delta = F.linear(F.linear(x.float(), pair.a), pair.b) * pair.scaling
        result[..., start:end].add_(delta)
    return result.to(y.dtype)


def expert_residual(x: torch.Tensor, ids: torch.Tensor, pair: LoRAPair, *, routed: bool) -> torch.Tensor:
    """Return [routes, out] deltas indexed by logical experts, never cache slots.

    Bound the gathered weights during prefill; loop lengths depend only on shapes,
    so the same code is safe during decode CUDA graph capture.
    """
    flat = ids.reshape(-1).long()
    result = torch.empty((flat.numel(), pair.b.shape[1]), dtype=torch.float32, device=x.device)
    top_k = ids.shape[1]
    for start in range(0, flat.numel(), 256):
        end = min(start + 256, flat.numel())
        logical = flat[start:end]
        rows = torch.arange(start, end, device=x.device)
        inputs = x.index_select(0, rows if routed else rows // top_k).float()
        a = pair.a.index_select(0, logical)
        b = pair.b.index_select(0, logical)
        low = torch.bmm(a, inputs.unsqueeze(-1))
        result[start:end].copy_(torch.bmm(b, low).squeeze(-1) * pair.scaling)
    return result


def gate_up_residual(x, ids, pairs):
    return torch.cat([expert_residual(x, ids, pairs[name], routed=False) for name in ("gate", "up")], dim=-1)
