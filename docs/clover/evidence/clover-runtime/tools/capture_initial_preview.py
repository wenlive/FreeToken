"""Record identical early masked blocks for an independent HF numeric audit."""
import argparse
import json
from pathlib import Path

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompts-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prompts = json.loads(args.prompts_json.read_text(encoding='utf-8'))
    llm = LLM(args.model, moe_strategy='offload', moe_cache_size=2560,
        expert_load='serial', max_running_req=1, memory_ratio=.85,
        num_token_override=4096, max_seq_len_override=4096,
        max_extend_tokens=4096, cuda_graph_bs=[1])
    runtime = llm.engine.mtp_runtime
    original = runtime._preview
    result = dict(state='running', scope='First two previews of the first block; exact native inputs, same prompt history', cases=[])
    current = {}
    count = 0

    def observed(req, block):
        nonlocal count
        logits = original(req, block)
        if count < 2:
            filename = f"{current['prompt']}-W{current['width']}-step{count}.pt"
            torch.save(logits.detach().float().cpu().clone(), args.output / filename)
            result['cases'].append(dict(**current, step=count, block_ids=block.cpu().tolist(), logits_file=filename))
        count += 1
        return logits

    runtime._preview = observed
    try:
        for name, widths in [('Count', (32,)), ('JSON', (8, 16, 32))]:
            ids = prompts['prompts'][name]['prompt_ids']
            for width in widths:
                current = dict(prompt=name, width=width, prompt_ids=ids)
                count = 0
                output = llm.generate([ids], SamplingParams(temperature=0, max_tokens=width,
                    clover_mode='denoise', clover_width=width))[0]
                assert count >= 2
                result.setdefault('outputs', []).append(dict(prompt=name, width=width, text=output['text']))
                (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        result['state'] = 'complete'
        (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    finally:
        llm.engine.shutdown()


if __name__ == '__main__':
    main()
