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

## Capacity and sparse prefill

`clover_max_width=None` resolves to 0 for a causal engine, or `clover_width` for a
block engine. Explicit 0/8/16/32 remain supported. A causal engine serving request-level
block modes must declare their maximum width in advance. Exceeding it is an admission error.

`clover_sparse_prefill_limit` replaces `FREETOKEN_CLOVER_SPARSE_PREFILL`; the default
remains 128 rows, and 0 disables union admission. The threshold has not been tuned.
The Q8-only bank-descriptor and IQ4_XS changes are not part of this NVFP4 port.

## Pure causal overlap

Pure causal requests advance penalty counts from the consumed device token and the logical
input position. Delayed host delivery no longer decides whether to increment or reset
counts. Prompt tokens are excluded, and the runtime rejects a skipped response position.

The scheduler can overlap requests whose resolved mode is causal. Selfspec and denoise
requests retain their drain barrier even during causal prefill or tail fallback, so stop
strings and first-block delivery see complete history. Flash-Next MTP retains its barrier.
GPU/P1 overlap-on/off and throughput qualification are tracked separately from unit tests.
