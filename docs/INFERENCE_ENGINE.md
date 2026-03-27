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
  kernel, the local `qwen3.5:0.8b` benchmark on this host measured about
  `515 tok/s` decode on Psionic versus
  about `329 tok/s` decode on local Ollama for the same one-sentence prompt
  and `128` token cap.
- The qwen35 lane is now ahead on decode throughput for this host and prompt,
  but it is still not architecture-closed. Greedy prompt replay is materially
  faster than the earlier pilot, but the remaining headroom is still in the
  host-seeded hidden copy, output-head launch overhead, and a more integrated
  full-attention decode path.

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
