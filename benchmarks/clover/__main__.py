"""Clover measurement and reference tools; run from the repository root."""

import argparse
import runpy
import sys


TOOLS = {
    "bench": ("bench_clover", {"resources": "bench_clover_gpu_resources"}),
    "context": ("bench_clover_context", {}),
    "preview": ("bench_clover_preview", {
        "split-kv": "bench_clover_split_kv",
        "audit": "check_clover_preview",
        "graphs": "check_clover_graphs",
        "lifecycle": "check_clover_lifecycle",
        "verify": "check_clover_verify",
        "outputs": "check_clover_prompt_outputs",
    }),
    "costs": ("analyze_clover_budget", {"abab": "summarize_clover_abab"}),
    "hf-check": ("check_clover_hf_preview", {
        "denoise": "check_clover_hf_denoise",
        "selfspec": "check_clover_hf_selfspec",
        "quantization": "check_clover_quantization",
    }),
    "quality": ("eval_clover_quality", {"fetch-data": "fetch_clover_eval_data"}),
}


def main():
    variants = "\n".join(
        f"  {command}: {', '.join(choices)}"
        for command, (_, choices) in TOOLS.items() if choices
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Optional tool selectors after the command:\n" + variants
        + "\nOmit the selector to run the default tool. Each tool accepts --help.",
    )
    parser.add_argument("command", choices=TOOLS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module, choices = TOOLS[args.command]
    remaining = args.arguments
    if remaining and remaining[0] in choices:
        module = choices[remaining.pop(0)]
    sys.argv = [f"python -m benchmarks.clover {args.command}", *remaining]
    runpy.run_module(f"benchmarks.clover.{module}", run_name="__main__")


if __name__ == "__main__":
    main()
