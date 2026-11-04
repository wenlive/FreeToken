"""Summarize four chronological B/A/B/A processes without hiding process drift."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


METRICS = ('end_to_end_tokens_per_second', 'decode_tokens_per_second',
           'ttft_seconds', 'expert_cache_hit_rate', 'decode_expert_copy_payload_bytes')


def summarize(turns):
    if len(turns) != 4:
        raise ValueError('Expected chronological before/after/before/after processes')
    settings = turns[0]['settings']
    if any(turn['state'] != 'complete' for turn in turns):
        raise ValueError('ABAB process is incomplete')
    if any(turn['settings'] != settings for turn in turns[1:]):
        raise ValueError('ABAB settings differ')
    if any(turn['prompts'] != turns[0]['prompts'] for turn in turns[1:]):
        raise ValueError('ABAB input tokens differ')
    grouped = []
    for turn in turns:
        groups = {}
        warmups = {}
        for row in turn['runs']:
            key = (row['prompt'], row['mode'], row['width'])
            if row['warmup']:
                warmups[key] = warmups.get(key, 0) + 1
            else:
                groups.setdefault(key, []).append(row)
        if any(len(rows) != settings['repeats'] for rows in groups.values()):
            raise ValueError('Incomplete measured repetitions')
        if any(warmups.get(key,0) != settings['warmups'] for key in groups):
            raise ValueError('Incomplete warmup repetitions')
        grouped.append(groups)
    if any(group.keys() != grouped[0].keys() for group in grouped[1:]):
        raise ValueError('ABAB cases differ')
    cases = []
    for key in grouped[0]:
        trials = [group[key] for group in grouped]
        signatures = [{hashlib.sha256(json.dumps(row['token_ids']).encode()).hexdigest()
                       for row in rows} for rows in trials]
        case = dict(prompt=key[0], mode=key[1], width=key[2],
                    output_signatures_per_turn=[sorted(value) for value in signatures],
                    all_measured_output_ids_equal=len(set.union(*signatures)) == 1,
                    metrics={})
        for metric in METRICS:
            values = [[row[metric] for row in rows if row.get(metric) is not None]
                      for rows in trials]
            if not all(values):
                continue
            medians = [statistics.median(value) for value in values]
            case['metrics'][metric] = dict(
                process_medians=medians,
                before_median=statistics.median(values[0] + values[2]),
                after_median=statistics.median(values[1] + values[3]),
                after_over_before_pairs=[medians[i+1]/medians[i] if medians[i] else None
                                         for i in (0, 2)],
                measured_min_max_per_turn=[[min(value), max(value)] for value in values])
        cases.append(case)
    return dict(state='complete', settings=settings,
        aggregation=f"Two independent processes per side; pooled median of {2*settings['repeats']} measured requests. "
        'Both chronological A/B process ratios and each five-sample range are retained. '
        'Warmups excluded. Ratios are after/before for every metric, including latency.',
        startup_seconds_per_turn=[turn['startup_seconds'] for turn in turns], cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turns', type=Path, nargs=4, required=True,
                        help='Chronological before/after/before/after results.json paths')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(path.read_text(encoding='utf-8')) for path in args.turns])
    result['inputs'] = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        for path in args.turns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(cases=len(result['cases']),
                         equal_outputs=sum(row['all_measured_output_ids_equal'] for row in result['cases']))))


if __name__ == '__main__':
    main()
