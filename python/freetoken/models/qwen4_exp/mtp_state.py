"""Capture the target residual before its final HC mix, with normal slot ownership."""
from __future__ import annotations

import torch

MTP_LAST_RESIDUAL = 'mtp_last_residual'


def capture_residual(residual, batch, pool, buffer):
    if residual.dtype != torch.float32 or residual.ndim != 2 or len(residual) > len(buffer):
        raise ValueError('MTP target residual exceeds its declared FP32 capture buffer')
    fla = batch.fla_metadata
    if fla is None:
        raise ValueError('MTP residual capture requires prepared FLA metadata')
    buffer[:len(residual)].copy_(residual)
    states = pool.slot_state(MTP_LAST_RESIDUAL)
    ends = fla.cu_seqlens[1:].long() - 1
    states.index_copy_(0, fla.cache_indices.long(), residual.index_select(0, ends))
    if fla.track_dst is not None:
        rows = fla.track_boundary_row.long() - 1
        states.index_copy_(0, fla.track_dst, residual.index_select(0, rows))
