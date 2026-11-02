"""Measure an unmodified official Clover llama.cpp build with exact prompt ids.

Server trials retain the expert cache but disable prompt/KV reuse. CLI trials
start a fresh model process: their cache state is deliberately labelled cold.
Raw streams/logs and the built-in EOS-inclusive counters are always preserved.
"""

import argparse
import json
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

STOP_IDS = {248044, 248046}


def cli_metrics(records, prompt):
    messages = [r for r in records if r.get("type") == "log"]
    text = "".join(r["msg"] for r in messages)
    match = re.search(r"encoded\s+(\d+) tokens in\s+([\d.]+) seconds", text)
    decoded = re.search(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds", text)
    if not match or not decoded:
        raise ValueError("Missing complete official CLI timing counters")
    count = int(match[1])
    pieces = [r for r in messages if r["level"] == "none"]
    if count != len(prompt["prompt_ids"]) or len(pieces) < count + 3:
        raise ValueError("CLI prompt token count differs from the fixed input")
    if pieces[0]["msg"] != "\n\n" or [r["msg"] for r in pieces[-2:]] != ["\n\n"] * 2:
        raise ValueError("Incomplete CLI output boundary")
    if "".join(r["msg"] for r in pieces[1 : count + 1]) != prompt["text"]:
        raise ValueError("CLI prompt echo differs from the fixed formatted prompt")
    visible = pieces[count + 1 : -2]
    start, end = pieces[count]["time"], pieces[-2]["time"]
    elapsed = (end - start) / 1e6
    if elapsed <= 0 or int(decoded[1]) < len(visible):
        raise ValueError("CLI producer timestamps must be enabled")
    return dict(
        text="".join(r["msg"] for r in visible),
        output_tokens=len(visible),
        wall_seconds=elapsed,
        ttft_seconds=(visible[0]["time"] - start) / 1e6 if visible else None,
        end_to_end_tokens_per_second=len(visible) / elapsed,
        builtin_predicted_tokens=int(decoded[1]),
        builtin_decode_seconds=float(decoded[2]),
        unprinted_builtin_tokens=int(decoded[1]) - len(visible),
        text_has_replacement_character=any("\ufffd" in r["msg"] for r in visible),
    )


def stream_metrics(events, wall):
    final = next((r["data"] for r in reversed(events) if r["data"].get("stop")), None)
    if final is None:
        raise ValueError("SSE stream has no terminal result")
    tokens, arrivals, text = [], [], []
    for event in events:
        data = event["data"]
        if "error" in data:
            raise ValueError(data["error"])
        current = data.get("tokens", [])
        if data.get("stop") and current[: len(tokens)] == tokens:
            # Some frontends repeat the full token list in the final packet.
            current = current[len(tokens) :]
        tokens.extend(current)
        arrivals.extend([event["seconds"]] * len(current))
        text.append(data.get("content", ""))
    stop_ids = STOP_IDS | (
        {tokens[-1]} if tokens and final.get("stop_type") == "eos" else set()
    )
    visible = [
        (token, at) for token, at in zip(tokens, arrivals) if token not in stop_ids
    ]
    return dict(
        text="".join(text),
        token_ids=[token for token, _ in visible],
        all_token_ids=tokens,
        output_tokens=len(visible),
        ttft_seconds=visible[0][1] if visible else None,
        wall_seconds=wall,
        end_to_end_tokens_per_second=len(visible) / wall,
        token_arrival_seconds=[at for _, at in visible],
        builtin_timings=final.get("timings", {}),
        effective_settings=final.get("generation_settings", {}),
        stop_type=final.get("stop_type"),
    )


def load_records(path):
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    return records


def request_json(url, data=None, timeout=2):
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def server_trials(command, prompt, args, directory):
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    command += [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        "1",
        "--sleep-idle-seconds",
        "1",
        "--no-warmup",
    ]
    base = f"http://127.0.0.1:{port}"
    with (
        (directory / "server.stdout.log").open("wb") as stdout,
        (directory / "server.stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            (directory / "command.json").write_text(
                json.dumps(command, indent=2), encoding="utf-8"
            )
            started = time.perf_counter()
            while time.perf_counter() - started < 120:
                if process.poll() is not None:
                    raise RuntimeError(f"Official server exited {process.returncode}")
                try:
                    if request_json(base + "/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.HTTPError):
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Official server startup timeout")
            encoded = request_json(
                base + "/tokenize",
                dict(content=prompt["text"], add_special=True, parse_special=True),
            )
            if encoded["tokens"] != prompt["prompt_ids"]:
                raise ValueError(
                    "Official tokenizer differs from the fixed HF prompt ids"
                )
            (directory / "tokenizer-check.json").write_text(
                json.dumps(encoded, indent=2), encoding="utf-8"
            )
            for repeat in range(args.warmups + args.repeats):
                body = dict(
                    prompt=prompt["prompt_ids"],
                    n_predict=args.tokens,
                    stream=True,
                    return_tokens=True,
                    cache_prompt=False,
                    temperature=0,
                    top_k=1,
                    seed=42,
                    repeat_penalty=1,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
                request = urllib.request.Request(
                    base + "/completion",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                events = []
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=300) as response:
                    for line in response:
                        if line.startswith(b"data: "):
                            data = json.loads(line[6:])
                            events.append(
                                dict(seconds=time.perf_counter() - started, data=data)
                            )
                wall = time.perf_counter() - started
                (directory / f"request-{repeat}.json").write_text(
                    json.dumps(
                        dict(request=body, events=events), indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
                row = stream_metrics(events, wall)
                timing = row["builtin_timings"]
                if timing.get("cache_n", 0) or timing.get("prompt_n") != len(
                    prompt["prompt_ids"]
                ):
                    raise ValueError(
                        "Server reused prompt/KV state or changed token inputs"
                    )
                yield repeat, row
            # Let stock sleep mode destroy the model while the logger is alive.
            deadline = time.perf_counter() + 8
            while time.perf_counter() < deadline:
                records = load_records(directory / "server.stdout.log")
                if any("hit rate =" in r.get("msg", "") for r in records):
                    break
                time.sleep(0.2)
        finally:
            process.terminate()
            process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend", choices=["server", "cli"], default="server")
    parser.add_argument("--modes", default="causal,selfspec:8,selfspec:16")
    parser.add_argument("--prompts", default="Count,Sky,Python,JSON,Chinese")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-mib", type=int, default=6656)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--load-mode", choices=["mmap", "none"], default="mmap")
    args = parser.parse_args()
    if args.frontend == "server" and any(
        mode.startswith("official") for mode in args.modes.split(",")
    ):
        parser.error(
            "The official entropy flags are only exposed by llama-speculative-simple, not llama-server"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))["prompts"]
    result = dict(
        state="running",
        frontend=args.frontend,
        settings=vars(args).copy(),
        protocol="greedy, no penalties; official mode uses entropy acceptance0.9/0.6",
        cache_scope="one resident server per prompt/mode"
        if args.frontend == "server"
        else "new model process per request",
        runs=[],
        summary=[],
        backend_receipts=[],
    )
    result["settings"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in result["settings"].items()
    }

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save()
    try:
        for name in args.prompts.split(","):
            prompt = prompts[name]
            for config in args.modes.split(","):
                mode, _, width = config.partition(":")
                width = int(width or 1)
                if mode not in ("causal", "selfspec", "official") or (
                    mode != "causal" and width not in (8, 16)
                ):
                    raise ValueError("Unsupported official llama.cpp mode")
                directory = args.output / f"{name}-{mode}-{width}"
                directory.mkdir()
                command = [
                    str(args.executable),
                    "-m",
                    str(args.model),
                    "-c",
                    str(args.context_size),
                    "-b",
                    str(args.batch_size),
                    "-ub",
                    str(args.ubatch_size),
                    "-ngl",
                    "99",
                    "--cpu-moe",
                    "--moe-cache-mib",
                    str(args.cache_mib),
                    "--load-mode",
                    args.load_mode,
                    "-t",
                    "8",
                    "-tb",
                    "8",
                    "-ctk",
                    "bf16",
                    "-ctv",
                    "bf16",
                    "--flash-attn",
                    "on",
                    "--temp",
                    "0",
                    "--top-k",
                    "1",
                    "--seed",
                    "42",
                    "--repeat-penalty",
                    "1",
                    "--presence-penalty",
                    "0",
                    "--frequency-penalty",
                    "0",
                    "--log-jsonl",
                    "--log-timestamps",
                    "-lv",
                    "4",
                    "--spec-type",
                    "none" if mode == "causal" else "draft-self",
                ]
                if mode != "causal":
                    command += ["--spec-draft-n-max", str(width - 1)]
                if mode == "official":
                    command += [
                        "--spec-accept-ratio",
                        "0.9",
                        "--spec-accept-delta",
                        "0.6",
                    ]
                if args.frontend == "server":
                    iterator = server_trials(command, prompt, args, directory)
                else:

                    def cli_trials():
                        # common/arg.cpp removes one trailing LF from --file.
                        # Keep the actual model input identical to the server/HF.
                        prompt_file = directory / "prompt.txt"
                        prompt_file.write_bytes((prompt["text"] + "\n").encode("utf-8"))
                        for repeat in range(args.warmups + args.repeats):
                            cmd = command + [
                                "-f",
                                str(prompt_file),
                                "-n",
                                str(args.tokens),
                                "--color",
                                "off",
                            ]
                            (directory / f"command-{repeat}.json").write_text(
                                json.dumps(cmd, indent=2), encoding="utf-8"
                            )
                            path = directory / f"request-{repeat}.stdout.log"
                            with (
                                path.open("wb") as out,
                                (directory / f"request-{repeat}.stderr.log").open(
                                    "wb"
                                ) as err,
                            ):
                                subprocess.run(
                                    cmd, stdout=out, stderr=err, check=True, timeout=600
                                )
                            yield repeat, cli_metrics(load_records(path), prompt)

                    iterator = cli_trials()
                rows = []
                for repeat, row in iterator:
                    row.update(
                        prompt=name,
                        mode=mode,
                        width=width,
                        repeat=repeat,
                        warmup=repeat < args.warmups,
                    )
                    result["runs"].append(row)
                    if not row["warmup"]:
                        rows.append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "prompt",
                                    "mode",
                                    "width",
                                    "repeat",
                                    "output_tokens",
                                    "wall_seconds",
                                    "ttft_seconds",
                                )
                            }
                        ),
                        flush=True,
                    )
                summary = dict(prompt=name, mode=mode, width=width)
                for key in (
                    "wall_seconds",
                    "ttft_seconds",
                    "end_to_end_tokens_per_second",
                ):
                    values = [r[key] for r in rows if r[key] is not None]
                    summary[key] = statistics.median(values) if values else None
                result["summary"].append(summary)
                for path in directory.glob("*.log"):
                    for record in load_records(path):
                        message = record.get("msg", "")
                        if (
                            "MoE cache size" in message
                            or " slots (" in message
                            or "hit rate =" in message
                        ):
                            result["backend_receipts"].append(
                                dict(
                                    prompt=name,
                                    mode=mode,
                                    width=width,
                                    source=str(path),
                                    message=message,
                                )
                            )
                save()
        result["state"] = "complete"
        save()
    except BaseException as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
