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
- The first `qwen35` lane now supports bounded sampled decode on native CUDA
  when the request stays inside the exact candidate-only envelope:
  - sampled decode or non-zero effective temperature
  - effective `top_k` available and `<= 128`
  - repeat, presence, and frequency penalties inactive
  - structured-output masking inactive
- The runtime sampling surface now also honors `min_p`, `typical_p`,
  `mirostat`, `mirostat_tau`, `mirostat_eta`, and request-level
  `repeat_last_n` in addition to the existing sampled controls.
- The generic OpenAI-compatible qwen35 request surface now forwards
  `top_k`, `top_p`, `min_p`, `typical_p`, `mirostat`, `mirostat_tau`,
  `mirostat_eta`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`,
  `frequency_penalty`, and `seed` on both `/v1/chat/completions` and
  `/v1/responses`.
- `repeat_last_n` follows the Ollama-compatible local sampler contract:
  - default `64`
  - `0` disables the penalty lookback window
  - `-1` expands the penalty window to the full available history
- `min_p` remains compatible with the bounded qwen35 CUDA sampled lane because
  Psionic applies it after exact top-k candidate selection on both the dense
  and bounded sampling paths.
- `typical_p` remains compatible with the bounded qwen35 CUDA sampled lane for
  the same reason.
- The local Ollama `qwen3.5` runner on this checkout does not expose the full
  Psionic sampler surface through the same active path. Its live
  apples-to-apples sampler contract is the one built by
  `sample.NewSampler(temperature, topK, topP, minP, seed, grammar)`.
- `mirostat` is now supported on the qwen35 runtime surface too, but it is
  still exact-via-fallback rather than fast-path. The current lane routes it
  through explicit `raw_logits` readback instead of the bounded candidate lane.
- Outside that envelope the qwen35 lane still falls back to explicit
  `raw_logits` readback instead of silently narrowing behavior.
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
  also measures about `244 tok/s` on the local `qwen3.5:2b` artifact versus
  about `205 tok/s` on local Ollama, about `167 tok/s` on the local
  `qwen3.5:4b` artifact versus about `142 tok/s` on local Ollama, and about
  `103 tok/s` on the local `qwen3.5:9b` artifact versus about `95 tok/s` on
  local Ollama.
- On March 28, 2026, after adding a native one-row CUDA top-k candidate output
  path and routing qwen35 sampled decode through `TopKCandidates { top_k }`
  instead of unconditional dense-vocab readback, and after refreshing the
  local sampler surface to honor `min_p`, `typical_p`, `mirostat`,
  `mirostat_tau`, `mirostat_eta`, and request-level `repeat_last_n`,
  the same host measured native qwen35 sampled decode ahead of local Ollama on
  all four rows under the explicit sampled contract in
  `docs/QWEN35_OLLAMA_COMPARISON.md`: about `496 tok/s` versus `329 tok/s` on
  `qwen3.5:0.8b`, about `244 tok/s` versus `203 tok/s` on `qwen3.5:2b`, about
  `173 tok/s` versus `140 tok/s` on `qwen3.5:4b`, and about `105 tok/s`
  versus `93 tok/s` on `qwen3.5:9b`.
- The same March 28, 2026 refresh widened the Psionic-side sampler and request
  surface to include `typical_p`, `mirostat`, `mirostat_tau`,
  `mirostat_eta`, and request-level `repeat_last_n`, but those controls are
  not part of the canonical Psionic-versus-Ollama matrix on this checkout
  because the local `ollamarunner` qwen3.5 path does not wire them through the
  same active sampler path.
- On March 28, 2026, after widening the shared-memory CUDA top-k fast path to
  the full bounded qwen35 envelope and then replacing the older one-row
  radix-sort route with a partitioned multi-block one-row candidate path, the
  same host also measured the larger bounded-candidate `top_k = 100` contract
  ahead on all four rows in `docs/QWEN35_OLLAMA_COMPARISON.md`: about
  `416 tok/s` versus `320 tok/s` on `qwen3.5:0.8b`, about `225 tok/s` versus
  `204 tok/s` on `qwen3.5:2b`, about `163 tok/s` versus `124 tok/s` on
  `qwen3.5:4b`, and about `101 tok/s` versus `93 tok/s` on `qwen3.5:9b`.
- The 4B row only became correct and faster after fixing the fused decode
  output head for mixed `Q4_K` and `Q6_K` weights. Greedy `ArgmaxOnly` decode
  now routes `Q6_K` output weights through `Q8_1` projection plus `argmax_f32`
  instead of falling back to the slower generic quantized matvec path.
- The 9B row also fits and runs natively on this 16 GB host. The only extra
  benchmark requirement is operational: unload Ollama's resident GPU caches
  before measuring Psionic, because Ollama keeps prior model weights live in
  VRAM.
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
