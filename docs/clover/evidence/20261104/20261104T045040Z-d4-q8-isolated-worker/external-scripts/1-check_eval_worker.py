"""Run the relocated isolated worker on one correct and one incorrect completion."""
import argparse
import json
from pathlib import Path
from benchmarks.clover.eval_clover_quality import grade_code

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
problem = dict(prompt='def increment(x):', entry_point='increment',
    test='def check(candidate):\n    assert candidate(2) == 3\n    assert candidate(-1) == 0')
rows = []
for expected, completion in [(True, '    return x + 1'), (False, '    return x - 1')]:
    passed, details = grade_code(problem, completion)
    rows.append(dict(expected=expected, passed=passed, details=details))
    assert passed == expected, details
(a.output / 'results.json').write_text(json.dumps(dict(state='passed', cases=rows), indent=2), encoding='utf-8')
