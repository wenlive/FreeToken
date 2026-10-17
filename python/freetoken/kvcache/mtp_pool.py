"""Draft-cell KV follows the main token on which that cell depends."""
from __future__ import annotations

import torch

MTP_KV_HEADS = 2
MTP_HEAD_DIM = 256
MTP_KV_BYTES_PER_TOKEN = 2 * MTP_KV_HEADS * MTP_HEAD_DIM * 4 + 1


class MtpKVState:
    def __init__(self, num_tokens: int, device: torch.device) -> None:
        self.device = torch.device(device)
        self.rebuild(num_tokens)

    def release(self) -> None:
        self.keys = None
        self.values = None
        self.valid = None

    def rebuild(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            raise ValueError('MTP auxiliary KV needs positive token capacity')
        self.release()
        self.keys = torch.zeros((num_tokens, MTP_KV_HEADS, MTP_HEAD_DIM),
                                dtype=torch.float32, device=self.device)
        self.values = torch.zeros_like(self.keys)
        self.valid = torch.zeros(num_tokens, dtype=torch.bool, device=self.device)

    @property
    def owned_bytes(self) -> int:
        if self.keys is None:
            return 0
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values, self.valid))

    def invalidate(self, physical_slots: torch.Tensor) -> None:
        self.valid.index_fill_(0, physical_slots.long(), False)

    def publish(self, next_token_slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> None:
        # Cell p depends on token p+1: its owner is that next token's physical
        # slot. Never publish recursive, unverified draft cells into shared pages.
        shape = (next_token_slots.numel(), MTP_KV_HEADS, MTP_HEAD_DIM)
        if keys.shape != shape or values.shape != shape or keys.dtype != torch.float32 or values.dtype != torch.float32:
            raise ValueError('MTP auxiliary KV needs one FP32 key/value row per next-token slot')
        indices = next_token_slots.long()
        self.keys.index_copy_(0, indices, keys)
        self.values.index_copy_(0, indices, values)
        self.valid.index_fill_(0, indices, True)

    def restore_prefix(self, main_prefix_slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> int:
        # A P-token prefix owns only cells 0..P-2. The branch-dependent cell
        # P-1 must be recomputed with the new token using the restored last R.
        owners = main_prefix_slots[1:].long()
        count = owners.numel()
        if keys.shape != values.shape or keys.ndim != 3 or keys.shape[1:] != (MTP_KV_HEADS, MTP_HEAD_DIM):
            raise ValueError('MTP draft destination KV geometry differs')
        if count > keys.shape[0]:
            raise ValueError('MTP prefix exceeds private draft KV capacity')
        if count:
            # Once per request outside graph replay. Missing state is an error,
            # rather than allowing recycled physical pages to seed a draft.
            if not bool(self.valid.index_select(0, owners).all().item()):
                raise RuntimeError('MTP prefix contains unpublished auxiliary cells')
            keys[:count].copy_(self.keys.index_select(0, owners))
            values[:count].copy_(self.values.index_select(0, owners))
        return count
