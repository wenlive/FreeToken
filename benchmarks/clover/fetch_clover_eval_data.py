"""Fetch only fixed-revision OpenAI evaluation datasets, never model files."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request


def read(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "FreeToken-Clover-evaluation"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for repo, path, label in [
        ("openai/grade-school-math", "grade_school_math/data/test.jsonl", "gsm8k"),
        ("openai/human-eval", "data/HumanEval.jsonl.gz", "humaneval"),
    ]:
        commit = json.loads(
            read(f"https://api.github.com/repos/{repo}/commits/master")
        )["sha"]
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = read(url)
        data = gzip.decompress(raw) if path.endswith(".gz") else raw
        rows = data.decode("utf-8").strip().splitlines()
        (args.output / f"{label}-original.jsonl").write_bytes(data)
        (args.output / f"{label}-first50.jsonl").write_text(
            "\n".join(rows[:50]) + "\n", encoding="utf-8"
        )
        license_data = read(
            f"https://raw.githubusercontent.com/{repo}/{commit}/LICENSE"
        )
        (args.output / f"{label}-LICENSE").write_bytes(license_data)
        sources.append(
            dict(
                repository=repo,
                commit=commit,
                path=path,
                url=url,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                decoded_sha256=hashlib.sha256(data).hexdigest(),
                source_rows=len(rows),
                selection="first 50, unshuffled",
                license_sha256=hashlib.sha256(license_data).hexdigest(),
            )
        )
    (args.output / "sources.json").write_text(
        json.dumps(sources, indent=2), encoding="utf-8"
    )
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
