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
