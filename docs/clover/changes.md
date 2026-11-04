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
