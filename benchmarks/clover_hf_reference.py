"""Run unmodified official Clover arithmetic with bounded expert-weight storage.

Reference: DiffuSpace/Clover-35B-A3B-Preview at
4b0402b65e523d0183a14c431955438714cb16ff/{modeling,configuration,sampling}_clover_moe.py.
This harness replaces only expert tensor indexing: each selected expert is read
or dequantized on demand. Official forward/generate functions remain unchanged.
It is a correctness oracle, not a serving-performance implementation.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

import safetensors
import torch


def official_modules(source: Path):
    package = "_clover_official_4b0402b6"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(source)]
        sys.modules[package] = module
    model = importlib.import_module(package + ".modeling_clover_moe")
    config = importlib.import_module(package + ".configuration_clover_moe")
    sampling = importlib.import_module(package + ".sampling_clover_moe")
    return model, config, sampling


class ExpertStorage:
    def __init__(
        self, checkpoint, *, device="cuda", cache_bytes=256 << 20, precision="nvfp4"
    ):
        self.root = Path(checkpoint)
        self.weight_map = json.loads(
            (self.root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )["weight_map"]
        self.config = json.loads(
            (self.root / "config.json").read_text(encoding="utf-8")
        )
        self.device, self.limit, self.precision = device, cache_bytes, precision
        self.cache = OrderedDict()
        self.handles = {}
        self.bytes = 0
        self.reads = self.hits = 0

    def read(self, name, selection=None):
        filename = self.weight_map[name]
        if filename not in self.handles:
            self.handles[filename] = safetensors.safe_open(
                self.root / filename, framework="pt", device="cpu"
            )
        handle = self.handles[filename]
        tensor = (
            handle.get_tensor(name)
            if selection is None
            else handle.get_slice(name)[selection]
        )
        return tensor.to(self.device, non_blocking=True)

    def projection(self, layer, expert, projection):
        key = layer, expert, projection
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.reads += 1
        if self.precision == "nvfp4":
            prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
            packed = self.read(prefix + ".weight")
            scale = self.read(prefix + ".weight_scale").float()
            glob = self.read(prefix + ".weight_scale_2").float()
            # Independent format decoder: adjacent low/high nibbles, group 16.
            lookup = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                device=self.device,
            )
            codes = torch.empty(
                (*packed.shape[:-1], packed.shape[-1] * 2),
                dtype=torch.long,
                device=self.device,
            )
            codes[..., 0::2], codes[..., 1::2] = (
                (packed & 15).long(),
                (packed >> 4).long(),
            )
            value = (lookup[codes] * scale.repeat_interleave(16, -1) * glob).bfloat16()
        else:
            down = projection == "down"
            name = f"model.layers.{layer}.mlp.experts." + (
                "down_proj" if down else "gate_up_proj"
            )
            width = self.config["moe_intermediate_size"]
            rows = (
                slice(None)
                if down
                else slice(0, width)
                if projection == "gate"
                else slice(width, 2 * width)
            )
            value = self.read(name, (expert, rows, slice(None)))
            if self.precision == "fp8":
                scale = (value.float().abs().amax(-1, keepdim=True) / 448.0).clamp_min(
                    1e-30
                )
                value = (
                    (value.float() / scale).to(torch.float8_e4m3fn).float() * scale
                ).bfloat16()
        count = value.numel() * value.element_size()
        while self.cache and self.bytes + count > self.limit:
            _, evicted = self.cache.popitem(last=False)
            self.bytes -= evicted.numel() * evicted.element_size()
        if count <= self.limit:
            self.cache[key] = value
            self.bytes += count
        return value


class ExpertIndex:
    def __init__(self, storage, layer, projection):
        self.storage, self.layer, self.projection = storage, layer, projection

    def __getitem__(self, expert):
        if self.projection == "gate_up":
            return torch.cat(
                [
                    self.storage.projection(self.layer, expert, p)
                    for p in ("gate", "up")
                ],
                dim=0,
            )
        return self.storage.projection(self.layer, expert, self.projection)


def load_official(
    source, checkpoint, *, device="cuda", cache_bytes=6 << 30, precision="nvfp4"
):
    source, checkpoint = Path(source), Path(checkpoint)
    model_code, config_code, _ = official_modules(source)
    raw_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    raw_config.pop("quantization_config", None)
    config = config_code.CloverMoeConfig(**raw_config)
    with torch.device("meta"):
        model = model_code.CloverMoeModelLM(config)
    storage = ExpertStorage(
        checkpoint, device=device, cache_bytes=cache_bytes, precision=precision
    )
    for index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        del experts.gate_up_proj
        del experts.down_proj
        experts.gate_up_proj = ExpertIndex(storage, index, "gate_up")
        experts.down_proj = ExpertIndex(storage, index, "down")
    parameters = dict(model.named_parameters())
    expected = {n for n in storage.weight_map if ".mlp.experts." not in n}
    if set(parameters) != expected:
        raise ValueError(
            f"Official parameter coverage differs: missing={set(parameters) - expected}, extra={expected - set(parameters)}"
        )
    for name in parameters:
        parent, leaf = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        setattr(
            module, leaf, torch.nn.Parameter(storage.read(name), requires_grad=False)
        )
    if any(t.is_meta for t in model.parameters()):
        raise RuntimeError("Unloaded official model parameters")
    model.eval()
    return model, storage


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision", choices=["nvfp4", "bf16", "fp8"], default="nvfp4"
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--checkpoint-penalties", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    model, storage = load_official(
        args.source, args.checkpoint, precision=args.precision
    )
    result = dict(
        state="running",
        purpose="official same-weight correctness reference; not a performance baseline",
        precision=args.precision,
        startup_seconds=time.perf_counter() - started,
        official_sources={
            name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
            for name in (
                "modeling_clover_moe.py",
                "configuration_clover_moe.py",
                "sampling_clover_moe.py",
            )
        },
        cases=[],
    )
    native = json.loads(args.native_results.read_text(encoding="utf-8"))
    cases = native.get("cases")
    if cases is None:
        cases = [
            dict(
                name=name,
                prompt_ids=value["token_ids"],
                token_ids=next(
                    row["token_ids"]
                    for row in native["runs"]
                    if row["prompt"] == name and row["mode"] == "causal"
                ),
            )
            for name, value in native["prompts"].items()
        ]
    _, _, sampling = official_modules(args.source)
    from types import SimpleNamespace

    gen = SimpleNamespace(
        **{
            **sampling.PENALTY_DEFAULTS,
            **json.loads(
                (args.checkpoint / "generation_config.json").read_text(encoding="utf-8")
            ),
        }
    )

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save()
    for case in cases:
        ids = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda")
        output = model(ids, use_cache=True)
        scores = output.logits[0, -1].float()
        cache = output.past_key_values
        state = sampling.CloverPenaltyState(
            gen, model.config.vocab_size, "cuda", gen.eos_token_id
        )

        def penalize(values):
            values = values.float().clone()
            if args.checkpoint_penalties:
                values[151669] = -float("inf")
                state.apply(
                    values.reshape(1, -1), torch.tensor([151669], device="cuda"), 151669
                )
            return values

        scores = penalize(scores)
        torch.save(scores.cpu(), args.output / f"{case['name']}-first-logits.pt")
        native_scores = torch.load(
            args.native_results.parent / f"{case['name']}-first-logits.pt",
            weights_only=True,
        ).to(scores.device)
        p = scores.log_softmax(-1)
        q = native_scores.float().log_softmax(-1)
        kl = float(torch.where(p.exp() > 0, p.exp() * (p - q), 0).sum())
        reference_ids, forced_ids = [], []
        for token in case["token_ids"][: args.steps]:
            reference_ids.append(int(scores.argmax()))
            forced_ids.append(token)
            state.commit(torch.tensor([token], device="cuda"))
            output = model(
                torch.tensor([[token]], device="cuda"),
                past_key_values=cache,
                use_cache=True,
            )
            scores, cache = penalize(output.logits[0, -1]), output.past_key_values
        row = dict(
            name=case["name"],
            first_kl_hf_to_native=kl,
            native_ids=forced_ids,
            hf_argmax_ids=reference_ids,
            argmax_agreement=sum(a == b for a, b in zip(forced_ids, reference_ids))
            / len(forced_ids),
        )
        result["cases"].append(row)
        save()
        print(json.dumps(row), flush=True)
    result.update(
        state="complete",
        expert_reads=storage.reads,
        expert_cache_hits=storage.hits,
        peak_allocated=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.perf_counter() - started,
    )
    save()


if __name__ == "__main__":
    main()
