from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, List, Mapping

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # --hf-overrides: applied to the checkpoint config the model is built from (cached_load_hf_config)
    hf_overrides: Mapping[str, Any] = field(default_factory=dict)
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    ple_row_cache_rows: int = 0
    # DeepSeek-V4.1 Decoder SWA Bounded Replay: "bounded" (default) runs the decoder layers on each
    # prompt's last window; "exact" runs them on every token.
    swa_decoder_replay: str = "bounded"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    # Explicit opt-in: GPU-only owners plus compact host complements, fixed for this load.
    moe_q2_resident_experts: int = 0
    moe_q2_dynamic_slots: int = 1024
    # 仅永久Q2所有者缓存：将缺失专家搬运与共享专家计算重叠。
    moe_q2_decode_overlap: bool = True
    # Online owner exchange is opt-in; zero retains the frozen initial residency.
    moe_q2_adapt_every: int = 0
    moe_q2_adapt_decay: float = 1.0
    gguf_embedding_host: bool = False
    moe_q2_prefill_workspace: bool = False
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    linear_state_cache_slots: int | None = None
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)
    # Independent draft weights; absent keeps the native no-MTP path and budget.
    mtp_runtime_path: str | None = None
    mtp_max_context: int = 8192
    mtp_compact_state: bool = False

    def __post_init__(self):
        if self.moe_q2_prefill_workspace and (not self.moe_q2_resident_experts or self.moe_q2_dynamic_slots != 1024):
            raise ValueError('Prefill workspace borrowing requires protected Q2 with 1024 disposable dynamic rows')
        if not 0 <= self.ple_row_cache_rows <= (1 << 24) or 0 < self.ple_row_cache_rows < 8:
            raise ValueError('PLE row cache needs zero or8..16777216 rows')
        if self.ple_row_cache_rows and self.ple_backend != 'disk':
            raise ValueError('PLE row caching requires the disk backend')
        if self.linear_state_cache_slots is not None and self.linear_state_cache_slots < 0:
            raise ValueError('Linear snapshot cache slots cannot be negative')
        if self.moe_q2_adapt_every < 0 or (self.moe_q2_adapt_every and not self.moe_q2_resident_experts):
            raise ValueError('Online Q2 adaptation requires protected owners and a nonnegative interval')
        if not 0 < self.moe_q2_adapt_decay <= 1:
            raise ValueError('Online Q2 heat decay must be in (0, 1]')
        if self.mtp_compact_state and self.mtp_runtime_path is None:
            raise ValueError('Compact transaction state requires MTP weights')
        if self.mtp_runtime_path is not None:
            if (not self.mtp_runtime_path or self.max_running_req != 1 or self.tp_info.size != 1
                    or self.use_dummy_weight or self.moe_strategy != 'offload' or self.moe_cpu_layers):
                raise ValueError('MTP requires real weights, single-rank GPU offload and max_running_req=1')
            if self.mtp_max_context < 64 or self.mtp_max_context % 64:
                raise ValueError('MTP context capacity must be positive and page-aligned')
        if self.moe_q2_resident_experts < 0:
            raise ValueError('moe_q2_resident_experts must be nonnegative')
        if self.moe_q2_resident_experts:
            if (self.moe_strategy != 'offload' or self.moe_backend is not None
                    or self.moe_cpu_layers or self.use_dummy_weight
                    or self.moe_cache_auto or self.moe_cache_rate is not None
                    or self.moe_cache_policy != 'lru' or self.max_running_req != 1):
                raise ValueError('Protected Q2 requires explicit GPU offload, fixed LRU capacity and max_running_req=1')
            if self.moe_q2_dynamic_slots < 0:
                raise ValueError('Protected Q2 dynamic slots must be nonnegative')
            expected = 1024 + self.moe_q2_dynamic_slots + self.moe_q2_resident_experts
            if self.moe_cache_size != expected:
                raise ValueError(f'Protected Q2 moe_cache_size must be {expected}, got {self.moe_cache_size}')
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path, self.hf_overrides)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        if self.mtp_runtime_path is not None:
            from freetoken.models.config import SlotStateSpec
            from freetoken.models.qwen4_exp.mtp_state import MTP_LAST_RESIDUAL
            if (model_config.qwen4_args is None or getattr(quant, 'dialect', None) != 'gguf_strata'
                    or self.active_encoders or model_config.hidden_size != 2560
                    or model_config.qwen4_args.hc_count != 4):
                raise ValueError('MTP currently requires text-only GGUF Strata Flash Next geometry')
            capture_rows = getattr(self, 'max_extend_tokens', 4096)
            if capture_rows <= 0:
                raise ValueError('MTP capture row capacity must be positive')
            model_config = replace(model_config,
                slot_states=(*model_config.slot_states,
                             SlotStateSpec(MTP_LAST_RESIDUAL, (10240,), dtype=torch.float32)),
                mtp_runtime_path=self.mtp_runtime_path, mtp_capture_rows=capture_rows)
        return replace(model_config, quant=quant)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.table_positions

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
