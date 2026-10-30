"""Audit full-model preview logits and hash all persistent KV/GDN storage."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.models.clover_moe.preview import make_preview_batch


def persistent_hashes(engine):
    tensors = dict(conv=engine.linear_state_pool.conv_states, recurrent=engine.linear_state_pool.recurrent_states)
    tensors.update(engine.linear_state_pool.slot_states)
    for layer in engine.model.model.layers.op_list:
        if not layer._is_linear:
            identifier = layer.self_attn.layer_id
            tensors[f"key-{identifier}"] = engine.kv_cache.k_cache(identifier)
            tensors[f"value-{identifier}"] = engine.kv_cache.v_cache(identifier)
    return {name: hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for name, tensor in tensors.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(args.model, moe_strategy="offload", moe_cache_size=4096, expert_load="serial",
              max_running_req=1, memory_ratio=.85, num_token_override=4096, max_seq_len_override=4096,
              max_extend_tokens=4096, cuda_graph_bs=[1], moe_collect_stats=True)
    prompt = llm.tokenizer.apply_chat_template([dict(role="user",content="Count from 1 to 10.")],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    result = dict(state="running", cases=[])
    def save(): (args.output/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    original = llm.engine.forward_batch
    done = False
    def forward(batch, sampling):
        nonlocal done
        output = original(batch, sampling)
        if batch.is_prefill and not done:
            done = True
            for width in (8,16,32):
                before = persistent_hashes(llm.engine)
                block = [151669] * width
                preview = make_preview_batch(llm.engine, batch.reqs[0], block)
                with llm.engine.ctx.forward_batch(preview):
                    logits = llm.engine.model.forward()
                after = persistent_hashes(llm.engine)
                torch.save(logits.detach().float().cpu(), args.output/f"preview-{width}-logits.pt")
                row = dict(width=width, prompt_ids=prompt, block_ids=block, before=before, after=after,
                           unchanged=before==after, argmax=logits.argmax(-1).cpu().tolist())
                result["cases"].append(row)
                save()
                print(json.dumps(dict(width=width,unchanged=row["unchanged"],argmax=row["argmax"])),flush=True)
                if before != after:
                    raise AssertionError("Preview mutated persistent KV/GDN bytes")
        return output
    llm.engine.forward_batch = forward
    try:
        result["continuation"] = llm.generate([prompt],SamplingParams(temperature=0,max_tokens=4))
        if len(result["cases"]) != 3:
            raise AssertionError("Missing preview audit cases")
        result["state"]="complete"
        save()
    finally:
        llm.engine.shutdown()


if __name__ == "__main__": main()
