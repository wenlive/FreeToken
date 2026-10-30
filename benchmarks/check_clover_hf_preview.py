"""Compare native read-only previews to the fixed official same-weight HF model."""
import argparse
import json
from pathlib import Path

import torch

from clover_hf_reference import load_official


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=['nvfp4','bf16','fp8'],default='nvfp4')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, storage = load_official(args.source, args.checkpoint,precision=args.precision)
    native = json.loads(args.native_results.read_text(encoding="utf-8"))
    result = dict(state="running", precision=args.precision,cases=[])
    cache = None
    for case in native["cases"]:
        if cache is None:
            cache = model.model(torch.tensor([case["prompt_ids"]], device="cuda"), use_cache=True).past_key_values
        output = model(torch.tensor([case["block_ids"]], device="cuda"), past_key_values=cache, use_cache=True, preview=True)
        scores = output.logits[0].float()
        actual = torch.load(args.native_results.parent / f'preview-{case["width"]}-logits.pt', weights_only=True).cuda()
        p, q = scores.log_softmax(-1), actual.log_softmax(-1)
        expected_ids, actual_ids = scores.argmax(-1), actual.argmax(-1)
        expected_confidence = p.exp().amax(-1)
        actual_confidence = q.exp().amax(-1)
        torch.save(scores.cpu(), args.output/f'preview-{case["width"]}-logits.pt')
        row = dict(width=case["width"], mean_kl=float((p.exp()*(p-q)).sum(-1).mean()),
                   argmax_agreement=float((expected_ids==actual_ids).float().mean()),
                   hf_ids=expected_ids.tolist(), native_ids=actual_ids.tolist(),
                   hf_above_095=int((expected_confidence>.95).sum()), native_above_095=int((actual_confidence>.95).sum()))
        result["cases"].append(row)
        print(json.dumps(row),flush=True)
        (args.output/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    result["state"]="complete"
    (args.output/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
