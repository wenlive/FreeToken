"""Bounded subprocess grader for the fixed HumanEval test cases."""

import ast
import builtins
import json
from pathlib import Path
import sys

ALLOWED_IMPORTS = {
    "math",
    "re",
    "itertools",
    "collections",
    "typing",
    "functools",
    "statistics",
    "string",
    "heapq",
    "bisect",
    "operator",
    "decimal",
    "fractions",
    "random",
    "numpy",
    "copy",
}
FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
    "__import__",
}
FORBIDDEN_ATTRIBUTES = {
    "save",
    "load",
    "tofile",
    "dump",
    "read",
    "write",
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "chdir",
    "mkdir",
    "makedirs",
}


def main():
    case = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    code = (
        case["prompt"]
        + "\n"
        + case["completion"]
        + "\n"
        + case["test"]
        + "\ncheck("
        + case["entry_point"]
        + ")\n"
    )
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            n.name.split(".")[0] not in ALLOWED_IMPORTS for n in node.names
        ):
            raise ValueError("import outside the evaluation allowlist")
        if isinstance(node, ast.ImportFrom) and (
            node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS
        ):
            raise ValueError("import outside the evaluation allowlist")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            raise ValueError("unsupported evaluation builtin")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES
        ):
            raise ValueError("unsupported evaluation attribute")
    namespace = dict(vars(builtins))
    original_import = builtins.__import__

    def safe_import(name, *args, **kwargs):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    for name in FORBIDDEN_CALLS:
        namespace.pop(name, None)
    namespace["__import__"] = safe_import
    exec(
        compile(tree, "<humaneval>", "exec"),
        {"__builtins__": namespace, "__name__": "__main__"},
    )
    print("PASS")


if __name__ == "__main__":
    main()
