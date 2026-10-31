"""Clover-only short-prompt expert-union admission, with an A/B switch."""

import os

import torch

from freetoken.core import get_global_ctx
from freetoken.moe.fused import fused_topk
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE


def admit_expert_union(cache, layer_id, indices):
    """Compact raw routes to E carriers, admit once, and gather row slots."""
    count = torch.zeros(cache.num_experts, device=indices.device, dtype=torch.int32)
    flat = indices.reshape(-1).long()
    count.scatter_add_(0, flat, torch.ones_like(flat, dtype=count.dtype))
    experts = torch.arange(len(count), device=indices.device, dtype=indices.dtype)
    carrier = torch.where(count > 0, experts, indices.flatten()[0]).reshape(
        -1, indices.shape[-1]
    )
    cache.ensure_experts(layer_id, carrier)
    cache.copy_missing()
    return carrier.flatten()[indices.long()]


class CloverMoE(Qwen3_5MoE):
    def __init__(self, config, layer_id=None, *, prefix=""):
        super().__init__(config, layer_id, prefix=prefix)
        self.sparse_prefill_limit = (
            128 if os.environ.get("FREETOKEN_CLOVER_SPARSE_PREFILL", "1") == "1" else 0
        )

    def forward(self, hidden_states):
        cache = getattr(self.experts, "offload_cache", None)
        if not (
            get_global_ctx().batch.is_prefill
            and cache is not None
            and cache.quant_format == "nvfp4"
            and cache.decode_target == "gpu"
            and cache.cache_policy == "lru"
            and not cache.is_cpu_layer(self.experts.layer_id)
            and hidden_states.shape[0] <= self.sparse_prefill_limit
        ):
            return super().forward(hidden_states)
        # Existing NVFP4 prefill streams all 256 experts. Short desktop prompts
        # can instead reuse the verified decode union/cache and unchanged GEMM.
        router = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states) * torch.sigmoid(
            self.shared_expert_gate.forward(hidden_states)
        )
        weights, indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router,
            topk=self.experts.top_k,
            renormalize=self.experts.renormalize,
        )
        # Admit the complete expert union with a fixed E-element carrier. Feeding
        # N*top_k assignments to the LRU caused multi-minute LLVM compilation for
        # N>32 on SM120. Duplicate the first active id in unused carrier positions;
        # it admits no additional experts and preserves the existing LRU kernel.
        slots = admit_expert_union(cache, self.experts.layer_id, indices)
        routed = self.experts._expert_gemm(
            cache,
            hidden_states,
            weights,
            slots,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.experts.layer_id),
            is_prefill=False,
        )
        return self.experts._maybe_all_reduce(routed) + shared
