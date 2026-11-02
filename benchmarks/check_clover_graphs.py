"""Audit real Clover graph replay, read-only state, changing prefixes and CUDA timing."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.models.clover_moe.preview import make_preview_batch
from check_clover_preview import persistent_hashes


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-slots", type=int, default=4096)
    parser.add_argument("--memory-ratio", type=float, default=.85)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(args.model, moe_strategy="offload", moe_cache_size=args.cache_slots,
        expert_load="serial", max_running_req=1, memory_ratio=args.memory_ratio, num_token_override=4096,
        max_seq_len_override=4096, max_extend_tokens=4096, cuda_graph_bs=[1], moe_collect_stats=True)
    engine, result = llm.engine, dict(state="running", previews=[], verifies=[], causal_ms=[])
    runtime = engine.mtp_runtime
    original = engine.forward_batch

    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def audited(batch, sampling):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(engine.stream)
        output = original(batch, sampling)
        end.record(engine.stream)
        end.synchronize()
        if batch.is_decode:
            result["causal_ms"].append(start.elapsed_time(end))
        if batch.is_prefill:
            req = batch.reqs[0]
            for width in (8, 16, 32):
                tokens = torch.full((width,), runtime.mask_id, device=engine.device, dtype=torch.int32)
                frozen = persistent_hashes(engine)
                eager = make_preview_batch(engine, req, tokens, coreset_top_p=engine.config.clover_coreset_top_p)
                with engine.ctx.forward_batch(eager):
                    reference = engine.model.forward().float().cpu()
                runtime._blocks().prepare_clean(req)
                actual = runtime._preview(req, tokens)
                torch.testing.assert_close(actual.float().cpu(), reference, rtol=0, atol=0)
                engine.moe_offload_cache.reset()
                engine.moe_offload_cache.reset_stats()
                actual = runtime._preview(req, tokens)
                torch.testing.assert_close(actual.float().cpu(), reference, rtol=0, atol=0)
                cold_stats = engine.moe_offload_cache.decode_miss_stats()
                static_batch = runtime._blocks().entries[(True, width, engine.config.clover_coreset_top_p)][0]
                updates = static_batch._clover_metadata_updates
                after = persistent_hashes(engine)
                if frozen != after:
                    raise AssertionError("Preview capture or replay changed persistent KV/GDN bytes")
                times = []
                for _ in range(7):
                    begin, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(engine.stream)
                    runtime._preview(req, tokens)
                    done.record(engine.stream)
                    done.synchronize()
                    times.append(begin.elapsed_time(done))
                assert updates == static_batch._clover_metadata_updates
                result["previews"].append(dict(prefix=req.cached_len, width=width,
                    hashes_before=frozen, hashes_after=after, logits_bitwise_equal=True, cold_stats=cold_stats,
                    warmups=2, repeats=5, milliseconds=times[2:], median_ms=statistics.median(times[2:])))
                save()
            # Compare the actual private transaction at every possible commit row.
            llm.cache_manager.allocate_paged([req], reserve_tokens=31)
            for width in (8, 16, 32):
                tokens = torch.full((width,), 64, device=engine.device, dtype=torch.int32)
                tokens[:1].copy_(output.next_tokens_gpu[:1])
                frozen_rec = digest(engine.linear_state_pool.recurrent_states)
                frozen_conv = digest(engine.linear_state_pool.conv_states)
                runtime._blocks().prepare_clean(req)
                if runtime.transactions is not None and runtime.transactions.capacity < width:
                    runtime._blocks().clear_verify()
                    runtime.transactions = None
                eager = runtime._verify_batch(req, tokens, capacity=width)
                with engine.ctx.forward_batch(eager):
                    reference = engine.model.forward().float().cpu()
                state_hashes = [(digest(runtime.transactions.recurrent[:, row]),
                                 digest(runtime.transactions.conv[:, row])) for row in range(width)]
                actual = runtime._verify(req, tokens, width)
                captured = actual.float().cpu()
                static_batch = runtime._blocks().entries[(False, width, 1.)][0]
                with engine.ctx.forward_batch(static_batch):
                    static_eager = engine.model.forward().float().cpu()
                diagnostics = dict(width=width, live_vs_static=float((reference - static_eager).norm() / reference.norm()),
                    static_vs_graph=float((captured - static_eager).norm() / static_eager.norm()),
                    live_vs_graph=float((captured - reference).norm() / reference.norm()))
                (args.output / f"verify-{req.cached_len}-{width}-diagnostic.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
                print(json.dumps(diagnostics), flush=True)
                torch.testing.assert_close(captured, reference, rtol=0, atol=0)
                replay_hashes = [(digest(runtime.transactions.recurrent[:, row]),
                                  digest(runtime.transactions.conv[:, row])) for row in range(width)]
                if replay_hashes != state_hashes:
                    raise AssertionError("Graph transaction states differ from eager")
                if frozen_rec != digest(engine.linear_state_pool.recurrent_states) or frozen_conv != digest(engine.linear_state_pool.conv_states):
                    raise AssertionError("Private verify graph changed live recurrent state")
                result["verifies"].append(dict(prefix=req.cached_len, width=width,
                    logits_bitwise_equal=True, state_hashes=state_hashes, live_state_unchanged=True))
                if hasattr(runtime.transactions, 'valid_width'):
                    keys_before = set(runtime._blocks().entries)
                    tails = []
                    for valid in range(1, width + 1):
                        begin = time.perf_counter()
                        tail = runtime._verify(req, tokens[:valid], width).float().cpu()
                        assert set(runtime._blocks().entries) == keys_before
                        assert runtime.transactions.valid_width == valid
                        torch.testing.assert_close(tail, reference[:valid], rtol=0, atol=0)
                        hashes = (digest(runtime.transactions.recurrent[:, valid - 1]),
                                  digest(runtime.transactions.conv[:, valid - 1]))
                        assert hashes == state_hashes[valid - 1]
                        tails.append(dict(valid_rows=valid, new_capture=False,
                            logits_bitwise_equal=True, state_bitwise_equal=True,
                            audit_wall_seconds=time.perf_counter()-begin))
                    result.setdefault('tails', []).append(dict(prefix=req.cached_len, width=width, cases=tails))
                eager = None
                save()
        return output

    engine.forward_batch = audited
    try:
        for text in ("Count from 1 to 10.", "Explain this in one sentence: " + "Trees shade streets. " * 40):
            ids = llm.tokenizer.apply_chat_template([dict(role="user", content=text)], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            llm.generate([ids], SamplingParams(temperature=0, max_tokens=64, clover_mode="causal"))
        result.update(state="complete", causal_median_ms=statistics.median(result["causal_ms"]))
        save()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
