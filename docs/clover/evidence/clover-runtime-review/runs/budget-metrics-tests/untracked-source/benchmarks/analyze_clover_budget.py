"""Conditional denoise throughput budgets from HF progress and native measured costs.

This is a fixed-hardware, fixed-prefix-distribution compute budget, not a universal
algorithmic upper bound. Native progress and controller efficiency are reported
separately; exceeding the HF-conditioned estimate is possible when progress differs.
"""
import argparse
import json
from pathlib import Path
import statistics


def analyze(native, hf):
    references = {(r['prompt'],r['width']):r for r in hf['cases']}
    measured = [r for r in native['runs'] if not r['warmup']]
    causal = {}
    for r in measured:
        if r['mode'] == 'causal':
            values = [t['cuda_ms'] for t in r['runtime_breakdown'] if t['phase']=='causal']
            causal.setdefault(r['prompt'],[]).extend(values)
    rows = []
    for key, reference in references.items():
        prompt, width = key
        samples = [r for r in measured if (r['prompt'],r['mode'],r['width']) == (prompt,'denoise',width)]
        if not samples:
            continue
        hf_tokens = len(reference['visible_token_ids'])
        hf_previews = sum(map(len,reference['blocks']))
        hf_blocks = len(reference['blocks'])
        trials = []
        for r in samples:
            trace = r['runtime_breakdown']
            count = lambda phase: sum(t['phase']==phase for t in trace)
            total = lambda phase: sum(t['cuda_ms'] for t in trace if t['phase']==phase)
            previews, blocks = count('preview'), count('verify')
            preview = total('preview')/previews
            commit_total = total('verify')+total('state_commit')+total('save_ahead')
            commit = commit_total/blocks
            compute = total('preview')+commit_total
            generation = total('round')-total('advance')
            hf_budget = hf_tokens*1000/(hf_previews*preview+hf_blocks*commit)
            observed = r['output_tokens']*1000/generation
            own_budget = r['output_tokens']*1000/compute
            trials.append(dict(preview_ms=preview,commit_ms=commit,
                native_visible_tokens=r['output_tokens'],native_previews=previews,native_blocks=blocks,
                native_useful_tokens_per_preview=r['output_tokens']/previews,
                hf_conditioned_tokens_per_second=hf_budget,
                native_progress_compute_tokens_per_second=own_budget,
                generation_envelope_tokens_per_second=observed,
                generation_envelope_ms=generation,compute_envelope_ms=compute,
                sampling_control_residual_ms=generation-compute,
                native_compute_efficiency=compute/generation,
                hf_conditioned_attainment=observed/hf_budget,
                steady_decode_tokens_per_second=r['decode_tokens_per_second'],
                end_to_end_tokens_per_second=r['end_to_end_tokens_per_second'],
                ttft_seconds=r['ttft_seconds'], first_burst_tokens=r['first_burst_tokens'],
                output_sha256=r.get('output_sha256')))
        summary = {k:statistics.median([r[k] for r in trials if r[k] is not None])
            for k in trials[0] if isinstance(trials[0][k],(int,float))}
        causal_step = statistics.median(causal[prompt]) if prompt in causal else None
        denominator = causal_step-summary['commit_ms']/width if causal_step is not None else None
        rows.append(dict(prompt=prompt,width=width,
            hf_visible_tokens=hf_tokens,hf_previews=hf_previews,hf_blocks=hf_blocks,
            hf_useful_tokens_per_preview=hf_tokens/hf_previews,
            hf_resolved_slots_per_preview=width*hf_blocks/hf_previews,
            causal_step_ms=causal_step,
            full_block_break_even_tokens_per_preview=(summary['preview_ms']/denominator
                if denominator is not None and denominator>0 else None),
            medians=summary,trials=trials))
    return dict(state='complete',scope=__doc__,
        formulas=dict(hf_conditioned='N_HF / (P_HF * mean_preview_seconds + B_HF * mean_commit_seconds)',
            native_progress='N_native / (sum_preview_seconds + sum_commit_seconds)',
            observed_generation='N_native / (sum_round_event_seconds - sum_advance_event_seconds)',
            full_block_break_even='mean_preview_seconds / (causal_step_seconds - mean_commit_seconds / W)'),
        boundaries=['Visible output excludes EOS and discarded resolved tail positions.',
            'Commit includes causal verification, live-state commit and saved-ahead copy.',
            'Round CUDA-event envelopes include GPU idle gaps caused by host control; '
            'scheduler gaps between rounds and prompt advance are excluded.',
            'CUDA-event call envelopes are not individual kernel-only timings.',
            'Native compute efficiency uses native progress; HF attainment may exceed 100% when progress differs.',
            'End-to-end and arrival-based steady decode metrics retain their original separate definitions.',
            'Averaged native costs are reused for the HF rollout as a counterfactual; route/cache/context distributions may differ.'],
        native_settings=native['settings'],hf_settings=hf.get('settings'),cases=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--hf',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    native=json.loads(a.native.read_text(encoding='utf-8'))
    hf=json.loads(a.hf.read_text(encoding='utf-8'))
    hf=hf.get('hf_denoise',hf)
    result=analyze(native,hf)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(state='complete',cases=len(result['cases']))))


if __name__=='__main__':
    main()
