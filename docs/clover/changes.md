# Clover repair notes

## Request admission

Clover resolves request mode, width and threshold before admitting the request. Invalid
settings, unsupported block backends and non-greedy block sampling return a request error;
they do not terminate the scheduler. Defaults are copied per request, so a shared
SamplingParams object is not mutated.

Denoising always previews its full declared width, including a short final delivery.
Admission therefore requires `prompt_tokens + ceil(max_tokens / width) * width <= max_seq_len`
after the scheduler's normal max_tokens clipping. A request that cannot fit is rejected
before allocating KV pages or linear state. Selfspec keeps its existing causal tail fallback.

Offline `LLM.generate` returns `text` and `token_ids` as before. A rejected item additionally
contains `error` and, when supplied by the scheduler, `error_code`; other items finish normally.

Implementation/qualification evidence is tracked in
`C:/workspace/tasks/T-20261104-clover-improve/README.md` until the phase evidence is archived.

## IQ4_XS expert strides

IQ4_XS now uses the actual distance between expert slots, like Q4_0/Q8_0/Q4_K/Q6_K.
This repairs mixed-layer banks with slot padding; official homogeneous Q8 is unaffected.
IQ4_XS support guarantees the addressing behavior, not grouped MMQ performance: when
either projection of a layer uses type 23, that layer's two projections use MMVQ even
for multiple input rows. Other layers using supported types can still use MMQ.

## Capacity and sparse prefill

`clover_max_width=None` now resolves to 0 for a causal engine, or `clover_width` for a
block engine. Explicit 0/8/16/32 remain supported. To serve block requests on an engine
whose default mode is causal, declare their largest width at engine construction.
Requests exceeding that declaration are rejected at admission.

For the official Q8 checkpoint, the private clean state plus transaction reserve is:

| Maximum width | Reserve bytes | Equivalent Q8 slots (3,342,336 bytes each) |
|---|---:|---:|
| 0 | 0 | 0 |
| 8 | 579,502,085 | 173.38 |
| 16 | 1,094,615,045 | 327.50 |
| 32 | 2,124,840,965 | 635.74 |

These are budget calculations; actual auto-cache capacity also depends on free memory,
other reservations and alignment. Explicit W32 retains the previous reservation.

`clover_sparse_prefill_limit` replaces `FREETOKEN_CLOVER_SPARSE_PREFILL`. It defaults to
128 rows; 0 disables union admission. The default threshold is unchanged pending TTFT
measurements. Mixed GGUF bank estimates require actual `expert_source.row_bytes` instead
of guessing Q8 for missing descriptors.

The shared encoder guard in EngineConfig is retained: frozen GGUF config shims have no
encoder fields, so assigning missing attributes to None fails. Present active encoder
sections still follow the existing disabling rules; the frozen-shim regression covers this.

## Pure causal overlap

Pure causal requests advance penalty counts from the consumed device token and the logical
input position. Delayed host delivery no longer decides whether to increment or reset
counts. Prompt tokens are excluded, and the runtime rejects a skipped response position.

The scheduler can overlap requests whose resolved mode is causal. Selfspec and denoise
requests retain their drain barrier even during causal prefill or tail fallback, so stop
strings and first-block delivery see complete history. Flash-Next MTP retains its barrier.
GPU/P1 overlap-on/off and throughput qualification are tracked separately from unit tests.

## Optional block graph warmup

`clover_graph_warmup_widths=[8,16,32]` captures preview and verify graphs before serving,
using the configured coreset top-p. Only list widths covered by `clover_max_width`.
The default is empty, so causal deployments pay no block capture cost. Warmup also runs
after an idle cache rebuild invalidates the graphs.

Warmup temporarily owns a request table, GDN slot and KV pages through the regular
allocators. It restores their contents and frees them without inserting a synthetic prefix.
Verify graphs share one transaction allocation sized for the largest declared width.
`runtime.graph_warmup` reports capture times and allocated/reserved memory deltas.
Cold-request latency and state-integrity qualification are pending; this is opt-in.
