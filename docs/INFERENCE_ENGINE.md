# Inference Engine

Psionic is only inference-ready when it can honestly serve compute products rather
than just run tensor math.

## Text Generation Requirements

- model load/unload lifecycle
- request execution path
- token streaming or equivalent delivery model
- KV cache lifecycle
- deterministic execution metadata
- runtime-side latency telemetry that keeps Tokio scheduling and async wait
  time separate from backend compute profiling
- backend capability gating
- served capability publication that keeps supported, route-required,
  refusal-required, and unsupported regions explicit together with context and
  latency envelopes

## Current Bounded Lanes

- Generic OpenAI-compatible GGUF serving may expose different runtime truth per
  loaded model inside the same process. Publication must stay model-specific in
  `/health`, `/v1/models`, and response headers.
- `qwen35` is `implemented_early` through a native Psionic CUDA text-generation
  runtime with prompt-projected image and video inputs at the HTTP layer.
- The `qwen35` lane must publish:
  - `backend = cuda`
  - `execution_mode = native`
  - `execution_engine = psionic`
  - `residency_mode = cuda_accelerated`
  - single-request execution posture
  - no scheduler policy claim
- The `qwen35` lane must also publish:
  - `multimodal_projection_mode = prompt_projection_only`
  - accepted projected media = `image`, `video`
  - the derived `qwen35` multimodal projection config from GGUF family facts
- The first `qwen35` lane supports prompt-replay response-state flows on
  `/v1/responses`.
- The first `qwen35` lane supports image and video request projection on
  `/v1/chat/completions` and `/v1/responses` without claiming a native image or
  video encoder.
- The first `qwen35` lane must fail closed for structured outputs and tool
  calling.
- The first `qwen35` lane must still fail closed for system-message image and
  video parts to stay aligned with the real template semantics.
- On March 27, 2026, after moving qwen35 hybrid-layer SSM `decay` and `beta`
  derivation onto CUDA, normalizing q/k regions directly into the packed
  decode buffers, replaying greedy argmax decode through a captured CUDA
  graph, and fusing the qwen35 full-attention query/gate split with per-head
  query RMSNorm, fusing per-head key RMSNorm with value packing, fusing the
  qwen35 dense and hybrid SiLU activation tails directly into GGML `Q8_1`
  scratch, and then fusing the full-attention sigmoid gating tail directly
  into GGML `Q8_1` scratch, and then replaying qwen35 prompt-prefix `NoOutput`
  submissions through a second captured CUDA graph, and then fusing qwen35
  hybrid q/k RMSNorm plus v staging into one CUDA kernel, and then fusing
  qwen35 hybrid depthwise conv1d plus immediate SiLU activation into one CUDA
  kernel, and then switching the q8_1 output-head argmax path onto the MMVQ
  kernel, and then switching the dense GGML `Q8_0` to `Q8_1` matvec fast path
  from the shared-input launcher onto the MMVQ kernel, and then retuning that
  q8.0 MMVQ launch from four warps per row down to two, and then broadcasting
  the q8.0 and q8.1 block scales once per four-lane MMVQ subgroup instead of
  rereading them on every lane, and then grouping the q8.0 output-head MMVQ
  argmax path into two-row blocks before the global argmax CAS, and then
  specializing the fully active dense q8.0 MMVQ decode path to use a fixed
  four-lane subgroup shuffle mask instead of recomputing `__activemask()` on
  every subgroup, the local `qwen3.5:0.8b` benchmark on this host measured
  about `523 tok/s` decode on Psionic versus about `329 tok/s` decode on local
  Ollama for the same one-sentence prompt and `128` token cap.
- On the same host, prompt, and token cap, the same native qwen35 CUDA lane now
  also measures about `247 tok/s` on the local `qwen3.5:2b` artifact versus
  about `203 tok/s` on local Ollama, and about `167 tok/s` on the local
  `qwen3.5:4b` artifact versus about `141 tok/s` on local Ollama.
- The 4B row only became correct and faster after fixing the fused decode
  output head for mixed `Q4_K` and `Q6_K` weights. Greedy `ArgmaxOnly` decode
  now routes `Q6_K` output weights through `Q8_1` projection plus `argmax_f32`
  instead of falling back to the slower generic quantized matvec path.
- The qwen35 lane is now ahead on decode throughput for this host and prompt,
  but it is still not architecture-closed. Greedy prompt replay is materially
  faster than the earlier pilot, but the remaining headroom is still in the
  host-seeded hidden copy, output-head launch overhead, and a more integrated
  full-attention decode path.
- The multi-row local comparison matrix for `0.8b`, `2b`, `4b`, and `9b` lives
  in `docs/QWEN35_OLLAMA_COMPARISON.md`.

## Embeddings Requirements

- explicit embeddings request/response contract
- deterministic vector shape metadata
- stable model identifier
- capability reporting tied to the served product
- execution receipt fields for outputs and runtime metadata

## KV Cache Requirements

Psionic now has served KV-cache support. The remaining completion bar is not
"whether KV cache exists." The remaining bar is whether the runtime can publish
truthful ownership, residency, reuse, and refusal behavior across host and
device paths.

The architecture must support:

- in-memory KV cache
- paged KV cache
- tiered KV cache
- concurrency-safe session ownership
- device-resident active decode state
- deferred host materialization for persistence, replay, and fallback paths

## Phase 0 Definition

Phase 0 is complete when Psionic can run a deterministic, CPU-backed
`psionic.embeddings` smoke path with truthful capability and receipt surfaces.
