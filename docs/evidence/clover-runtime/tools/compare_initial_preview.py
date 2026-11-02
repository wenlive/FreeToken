"""Compare fixed native masked blocks against the independent same-Q8 HF oracle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/workspace/wt/T-20261102-clover-q8/FreeToken/benchmarks')
import torch
from clover_hf_reference import load_official


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--native-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, storage = load_official(args.source, args.checkpoint, precision='gguf')
    native = json.loads(args.native_results.read_text(encoding='utf-8'))
    result = dict(state='running', scope='Forced identical prompt, masked block and empty response penalty history; not independent rollout', cases=[])
    previous, cache = None, None
    for case in native['cases']:
        if previous != case['prompt_ids']:
            previous = case['prompt_ids']
            cache = model.model(torch.tensor([previous], device='cuda'), use_cache=True).past_key_values
        expected = model(torch.tensor([case['block_ids']], device='cuda'), past_key_values=cache,
            use_cache=True, preview=True).logits[0].float()
        actual = torch.load(args.native_results.parent / case['logits_file'], weights_only=True).cuda()
        for scores in (expected, actual):
            scores[:, model.config.mask_token_id] = -float('inf')
        hp, np = expected.softmax(-1), actual.softmax(-1)
        hi, ni = expected.argmax(-1), actual.argmax(-1)
        hc, nc = hp.amax(-1), np.amax(-1)
        p, q = expected.log_softmax(-1), actual.log_softmax(-1)
        kl = torch.where(hp > 0, hp * (p-q), 0).sum(-1)
        row = dict(prompt=case['prompt'], width=case['width'], step=case['step'], block_ids=case['block_ids'],
            mean_kl=float(kl.mean()), argmax_agreement=float((hi==ni).float().mean()),
            hf_ids=hi.tolist(), native_ids=ni.tolist(), hf_confidence=hc.tolist(), native_confidence=nc.tolist(),
            threshold_crossing_rows=((hc>.95)!=(nc>.95)).nonzero().flatten().tolist(),
            hf_above_095=int((hc>.95).sum()), native_above_095=int((nc>.95).sum()))
        result['cases'].append(row)
        (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    result['state'] = 'complete'
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
