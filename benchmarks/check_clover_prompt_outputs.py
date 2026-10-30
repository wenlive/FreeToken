"""Check concrete code/JSON outputs separately from throughput and logit parity."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from eval_clover_quality import code_completion, grade_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Local tokenizer for HF case files containing token ids",
    )
    args = parser.parse_args()
    raw = args.results.read_bytes()
    source = json.loads(raw)
    result = dict(source_sha256=hashlib.sha256(raw).hexdigest(), cases=[])
    if "runs" in source:
        rows = [
            row
            for row in source["runs"]
            if not row["warmup"] and row["repeat"] == source["settings"]["warmups"]
        ]
    else:
        if args.tokenizer is None:
            parser.error("HF results require a local --tokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
        rows = [
            dict(
                **row,
                text=tokenizer.decode(row["token_ids"], skip_special_tokens=True),
                mode="denoise",
            )
            for row in source["cases"]
        ]
    for row in rows:
        text = row["text"]
        check = dict(prompt=row["prompt"], mode=row["mode"], width=row["width"])
        if row["prompt"] == "JSON":
            try:
                value = json.loads(text)
                books = value["books"]
                check["valid"] = len(books) == 3 and all(
                    set(book) == {"title", "author", "year", "tags"}
                    and isinstance(book["tags"], list)
                    and len(book["tags"]) == 2
                    and isinstance(book["year"], int)
                    and all(isinstance(book[key], str) for key in ("title", "author"))
                    for book in books
                )
            except (ValueError, KeyError, TypeError) as error:
                check.update(valid=False, error=type(error).__name__)
        elif row["prompt"] == "Python":
            try:
                code = code_completion(text)
                functions = [
                    node.name
                    for node in ast.parse(code).body
                    if isinstance(node, ast.FunctionDef)
                ]
                name = next(
                    (name for name in functions if "merge" in name), functions[0]
                )
                test = """def check(f):
    def normalize(x): return [tuple(pair) for pair in x]
    assert normalize(f([])) == []
    assert normalize(f([(1,3),(2,6),(8,10),(15,18)])) == [(1,6),(8,10),(15,18)]
    assert normalize(f([(8,10),(1,3),(2,6),(10,12)])) == [(1,6),(8,12)]
    assert normalize(f([(-10,-5),(-6,0),(-1,3)])) == [(-10,3)]
    assert normalize(f([(1,1),(1,2)])) == [(1,2)]
"""
                check["valid"], check["grade"] = grade_code(
                    dict(prompt="", test=test, entry_point=name), text
                )
            except (SyntaxError, IndexError, StopIteration) as error:
                check.update(valid=False, error=type(error).__name__)
        elif row["prompt"] == "Count":
            check["valid"] = list(map(int, re.findall(r"\d+", text))) == list(
                range(1, 11)
            )
        elif row["prompt"] == "Chinese":
            check["chinese_characters"] = len(re.findall(r"[\u4e00-\u9fff]", text))
            check["paragraphs"] = len(
                [p for p in re.split(r"\n\s*\n", text.strip()) if p]
            )
        result["cases"].append(check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
