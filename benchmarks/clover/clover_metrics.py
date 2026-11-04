"""Model-free token-arrival accounting for recorded Clover benchmarks."""

import statistics


def arrival_metrics(output_tokens, arrivals):
    if output_tokens < 0 or len(arrivals) not in (output_tokens, output_tokens + 1):
        raise ValueError("Arrival count must match output tokens plus optional EOS")
    # The scheduler also sends the terminal EOS event. Its latency and bundle
    # size must not inflate rates for the returned output token ids.
    visible = list(arrivals[:output_tokens])
    terminal = len(arrivals) - output_tokens
    if not visible:
        return dict(
            ttft_seconds=None,
            token_arrival_seconds=[],
            first_burst_tokens=0,
            decode_tokens_per_second=None,
            terminal_event_count=terminal,
        )
    first = sum(stamp == visible[0] for stamp in visible)
    return dict(
        ttft_seconds=visible[0],
        token_arrival_seconds=visible,
        first_burst_tokens=first,
        decode_tokens_per_second=(len(visible) - first) / (visible[-1] - visible[0])
        if visible[-1] > visible[0]
        else None,
        terminal_event_count=terminal,
    )


def corrected_timing_report(report):
    """Derive output-only timing from immutable historical event timestamps."""
    result = {**report, "runs": [], "summary": []}
    for row in report["runs"]:
        events = row.get("stream_event_arrival_seconds", row["token_arrival_seconds"])
        corrected = {
            **row,
            "stream_event_arrival_seconds": list(events),
            **arrival_metrics(row["output_tokens"], events),
        }
        previews = sum(len(r.get("steps", [])) for r in row["rounds"])
        corrected["visible_tokens_per_preview"] = (
            row["output_tokens"] / previews
            if row["mode"] == "denoise" and previews
            else None
        )
        result["runs"].append(corrected)
    for summary in report["summary"]:
        samples = [
            r
            for r in result["runs"]
            if not r["warmup"]
            and (r["prompt"], r["mode"], r["width"])
            == (summary["prompt"], summary["mode"], summary["width"])
        ]
        corrected = dict(summary)
        for key in (
            "ttft_seconds",
            "decode_tokens_per_second",
            "visible_tokens_per_preview",
        ):
            values = [r[key] for r in samples if r[key] is not None]
            corrected[key] = statistics.median(values) if values else None
        result["summary"].append(corrected)
    result["timing_definition"] = (
        "Output token ids, terminal EOS excluded; a single visible burst has no steady decode rate"
    )
    return result
