"""Real local GGUF/MTP compatibility, run once per mode and source checkout.

Use the same immutable model bundle and settings for main and the feature branch.
The script selects the requested source tree before importing FreeToken. It never
changes the model manifest, MTP qualification markers, installation or services.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mtp-runtime", type=Path, required=True)
    parser.add_argument("--mode", choices=["off", "on"], required=True)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "freetoken" in sys.modules:
        raise RuntimeError("Select the source tree before importing FreeToken")
    code = args.code_root.resolve(strict=True)
    sys.path.insert(0, str(code / "python"))
    import torch
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    source_file = Path(inspect.getfile(LLM)).resolve()
    if not source_file.is_relative_to(code):
        raise RuntimeError(f"Wrong FreeToken source: {source_file}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    enabled = args.mode == "on"
    resident = 4480 if enabled else 5248
    settings = dict(
        moe_strategy="offload",
        moe_cache_size=resident + 2048,
        moe_q2_resident_experts=resident,
        moe_q2_dynamic_slots=1024,
        attention_backend="qsa_sparse",
        ple_backend="disk",
        cache_type="radix",
        expert_load="serial",
        max_running_req=1,
        memory_ratio=0.9,
        num_token_override=8192,
        max_seq_len_override=8192,
        max_extend_tokens=1024,
        cuda_graph_bs=[1],
        linear_state_cache_slots=1,
        moe_q2_adapt_every=32,
        moe_q2_adapt_decay=0.7,
        gguf_embedding_host=True,
        ple_row_cache_rows=1048576,
        moe_q2_prefill_workspace=True,
        fixed_q2_rebuild_reserve_mib=0,
        mtp_runtime_path=str(args.mtp_runtime) if enabled else None,
        mtp_compact_state=enabled,
        mtp_max_context=8192,
        mtp_draft_policy="paired_cjk" if enabled else "full",
    )
    protected = [
        args.model / name
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
    ]
    protected += [
        args.mtp_runtime / "runtime.json",
        args.model.parents[1] / "manifest.json",
    ]
    before = {str(path): sha256(path) for path in protected}
    result = dict(
        state="running",
        mode=args.mode,
        model=str(args.model),
        settings=settings,
        engine_source=dict(
            root=str(code),
            llm_module=str(source_file),
            commit=subprocess.check_output(
                ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
            ).strip(),
        ),
        harness_sha256=sha256(Path(__file__)),
        protected_before=before,
        cases=[],
    )

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save()
    started = time.perf_counter()
    llm = LLM(str(args.model), **settings)
    runtime = llm.engine.mtp_runtime
    if enabled:
        assert type(runtime).__name__ == "MtpRuntime"
        assert Path(inspect.getfile(type(runtime))).resolve().is_relative_to(code)
        assert len(runtime.pool.reserved_slot_ids) == 4
        result["reserved_slots"] = [int(i) for i in runtime.pool.reserved_slot_ids]
        result["mtp_tensor_count"] = len(runtime.draft.metadata)
        assert result["mtp_tensor_count"] == 31
    else:
        assert runtime is None
    result["startup_seconds"] = time.perf_counter() - started
    prompts = dict(
        Arithmetic="计算17+25，只写结果。",
        Count="Count from 1 to 10.",
        Context="背景：这是兼容性测试，不需要总结。\n" * 16 + "请只回答 OK。",
    )
    try:
        for name, prompt in prompts.items():
            ids = llm.tokenizer.apply_chat_template(
                [dict(role="user", content=prompt)],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            if runtime is not None:
                runtime.rounds.clear()
            begun = time.perf_counter()
            answer = llm.generate([ids], SamplingParams(temperature=0, max_tokens=64))[
                0
            ]
            rounds = list(runtime.rounds) if runtime is not None else []
            assert answer["text"].strip() and answer["token_ids"]
            row = dict(
                prompt=name,
                prompt_ids=ids,
                **answer,
                rounds=rounds,
                seconds=time.perf_counter() - begun,
            )
            result["cases"].append(row)
            save()
            print(
                json.dumps(
                    dict(
                        prompt=name,
                        tokens=len(answer["token_ids"]),
                        mtp_rounds=len(rounds),
                        max_draft=max((r["draft_count"] for r in rounds), default=0),
                    )
                ),
                flush=True,
            )
        if enabled:
            assert any(
                r["draft_count"] > 0 for case in result["cases"] for r in case["rounds"]
            )
            assert all(
                0 <= r["draft_count"] <= 3
                for case in result["cases"]
                for r in case["rounds"]
            )
        after = {str(path): sha256(path) for path in protected}
        assert before == after
        result.update(
            state="complete",
            protected_after=after,
            elapsed_seconds=time.perf_counter() - started,
        )
        save()
    finally:
        llm.engine.shutdown()


if __name__ == "__main__":
    main()
