"""The scheduler's speculative runtime contract and lazy model dispatch."""

from typing import NamedTuple, Protocol


class DeliveryLimit(NamedTuple):
    count: int
    finished: bool


class SpeculativeRuntime(Protocol):
    def eligible(self, batch): ...
    def requires_drain(self, batch): ...
    def forward(self, batch, manager, token_pool, eos_ids, clip, anchor_id): ...
    def before_native(self, batch): ...
    def after_native(self, batch, native_state): ...
    def prepare_graphs(self): ...
    def close_graphs(self): ...


def create_runtime(engine):
    if engine.config.mtp_runtime_path is not None:
        from freetoken.models.qwen4_exp.mtp_runtime import MtpRuntime

        return MtpRuntime(engine, engine.model._mtp_drafter)
    if engine.config.model_config.model_type == "clover_moe":
        from freetoken.models.clover_moe.runtime import CloverRuntime

        return CloverRuntime(engine)
    return None


def runtime_reserve_bytes(config):
    """Price Clover's private clean state and largest permitted transaction."""
    if getattr(getattr(config, "model_config", None), "model_type", None) != "clover_moe":
        return 0
    width = config.clover_max_width
    if not width:
        return 0
    from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req, linear_state_dtypes

    conv_dtype, recurrent_dtype = linear_state_dtypes(config)
    row = linear_state_bytes_per_req(config.model_config.linear_attention_group(),
        config.tp_info.size, conv_dtype, recurrent_dtype=recurrent_dtype)
    return (width + 1) * row + 5
