"""Fresh-process adapter reload, shared-prefix requests, and runtime cache rebuild."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.mm.config import MultimodalConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-path")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    settings = dict(reference["settings"])
    if args.lora_path:
        settings["lora_path"] = args.lora_path
    elif "lora_path" in settings:
        raise ValueError("Reference is an adapter run; supply its artifact explicitly")
    with torch.inference_mode():
        start = time.perf_counter()
        llm = LLM(args.model, mm=MultimodalConfig(disabled_encoders=frozenset({"vision", "audio"})), **settings)
        result = {"status": "RUNNING", "reference": str(args.reference), "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  "load_seconds": time.perf_counter() - start, "adapter": llm.engine.lora_info, "checks": []}
        def save():
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        save()
        # Compare expert-only changes at the real model's first prefill activations.
        # A shallow operator copy suppresses only this call's expert residuals;
        # the live adapter, routing and cache banks are never modified.
        from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

        original_apply = TritonNvfp4MoEKernel.apply
        expert_audit = {}
        def audit_apply(kernel, layer, x, weights, ids, view, *, is_prefill):
            out = original_apply(kernel, layer, x, weights, ids, view, is_prefill=is_prefill)
            if is_prefill and layer.layer_id not in expert_audit and getattr(layer, "_lora_experts", None):
                base_layer = copy.copy(layer)
                base_layer._lora_experts = None
                base = original_apply(kernel, base_layer, x, weights, ids, view, is_prefill=is_prefill)
                delta = out.float() - base.float()
                expert_audit[layer.layer_id] = {"max_abs_expert_effect": delta.abs().max().item(),
                    "relative_l2_expert_effect": (delta.norm() / base.float().norm().clamp_min(1e-30)).item()}
            return out
        TritonNvfp4MoEKernel.apply = audit_apply
        def generate(record):
            tokens = llm.tokenizer.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False, return_dict=False)
            output = llm.generate([tokens], SamplingParams(temperature=0, max_tokens=32))[0]
            if output["token_ids"] != record["token_ids"]:
                raise AssertionError(f"Fresh/rebuilt engine output differs: {output}")
            return {"text": output["text"], "token_ids": output["token_ids"], "input_tokens": len(tokens)}
        # Reorder shared-prefix heldout prompts and interleave an unrelated replay.
        canary = reference["quality"]["canary-holdout.jsonl"]["records"]
        replay = reference["quality"]["replay.jsonl"]["records"]
        selected = [canary[0], canary[11], replay[0], canary[0], canary[49]]
        result["checks"].append({"kind": "fresh_reload_and_request_order", "outputs": [generate(row) for row in selected]})
        TritonNvfp4MoEKernel.apply = original_apply
        result["expert_effect_at_fixed_input_and_routes"] = expert_audit
        expected_layers = llm.engine.lora_info["expert_layers"] if llm.engine.lora_info else 0
        if len(expert_audit) != expected_layers:
            raise AssertionError("Did not observe every adapted expert layer")
        if args.lora_path:
            manifest = json.loads((Path(args.lora_path) / "manifest.json").read_text(encoding="utf-8"))
            zero_control = manifest.get("test_control", {}).get("kind") == "all-lora-B-zero"
            effects = [row["max_abs_expert_effect"] for row in expert_audit.values()]
            if (zero_control and any(value != 0 for value in effects)
                    or not zero_control and any(value <= 0 for value in effects)):
                raise AssertionError("Expert effect does not match the real/zero adapter control")
        save()
        identity = llm.engine.model.__dict__.get("_lora_identity")
        for size in (768, 1024):
            llm.rebuild_cache(moe_cache_size=size)
            if llm.engine.model.__dict__.get("_lora_identity") != identity:
                raise AssertionError("Adapter identity changed during expert cache rebuild")
            result["checks"].append({"kind": "expert_cache_rebuild", "slots": size, "outputs": [generate(row) for row in selected]})
            save()
        llm.rebuild_cache(num_pages=3072, num_mamba_slots=6)
        result["checks"].append({"kind": "kv_and_gdn_cache_rebuild", "pages": 3072, "usable_gdn_slots": 6,
                                  "outputs": [generate(row) for row in selected]})
        result["status"] = "PASS"
        save()


if __name__ == "__main__":
    main()
