"""The scheduler's speculative runtime contract and lazy model dispatch."""

from typing import NamedTuple, Protocol


class DeliveryLimit(NamedTuple):
    count: int
    finished: bool


class SpeculativeRuntime(Protocol):
    def eligible(self, batch): ...
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
