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
  - structured-output masking inactive
  - `mirostat` inactive
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
- Native qwen35 structured outputs are now supported on two explicit paths:
  - greedy, no-penalty, no-`mirostat` structured requests stay on
    `TopKCandidates { top_k: 128 }` and use exact sparse allowed-logit gather
    on candidate misses instead of dense vocab replay
  - structured requests outside that envelope still fall back to explicit
    `raw_logits` readback instead of silently narrowing behavior
- The native structured-output path now uses tokenizer-native incremental token
  append caches with per-leading-char token-id buckets and replay-safe sparse
  fallback, so qwen35 candidate misses no longer double-advance the decode
  state, linearly rescan the full vocabulary on sparse schema misses, or
  require dense vocab replay on the bounded structured lane.
- The local `qwen35_cuda_bench` harness now reproduces native-versus-Ollama
  JSON object and JSON schema requests too through `--json-object` and
  `--json-schema-file`, and it now writes machine-readable per-run evidence
  through `--json-out`.
- The repo-owned sequential collector for the canonical qwen35 versus Ollama
  matrix now lives at `scripts/release/run-qwen35-ollama-matrix.sh`. It writes
  a combined manifest plus row reports that preserve output-token arrays,
  prompt/decode timing, qwen35 output modes, readback bytes, raw-logit
  materialization, termination classification, first-divergence evidence, host
  power-limit metadata, Psionic commit, and Ollama version. The same harness
  now forces Psionic benchmark requests onto `PrefixCacheMode::Bypass` so the
  canonical matrix measures raw qwen35 prompt and decode throughput instead of
  unrelated shared prefix-cache behavior.
- Structured-output throughput is still not part of the canonical
  Psionic-versus-Ollama matrix. On March 28, 2026, after adding leading-char
  token-id buckets to the structured-output append cache, removing per-rule
  name allocation from the recursive matcher memo path, and rebuilding from a
  clean isolated target, a fresh local `qwen3.5:0.8b` summary-schema spot
  check measured native Psionic at about `78 tok/s` on the first bounded
  sparse-gather run and about `162 tok/s` mean across a warmed three-repeat
  pass, versus local Ollama at about `331 tok/s`. Psionic published
  `qwen35_output_modes=[top_k_candidates:128,sparse_logits:2,sparse_logits:3,sparse_logits:10]`,
  `qwen35_readback_bytes=5700`, and `qwen35_raw_logits=false` on the sparse
  run. The later warmed repeats stayed on `qwen35_output_modes=[top_k_candidates:128]`
  and hit the token cap without materializing `structured_output_value`, so
  this remains a bounded parity note rather than a canonical throughput row.
- Native `qwen35` tool calling is now supported on the generic
  OpenAI-compatible server surface through the bounded tagged-JSON-schema tool
  contract:
  - `/v1/chat/completions`
  - modes `none`, `auto`, `required`, and named tool choice
  - request-level `parallel_tool_calls`
  - ordered machine-readable `message.tool_calls`
  - streamed `delta.tool_calls` with ordered per-call indexes
  - `/v1/responses` prompt-replay state continuation across assistant tool
    turns and replayed `role = tool` results
  - JSON-schema-subset argument validation
  - proxy `qwen35` still fails closed for tool calling
- The first `qwen35` lane must still fail closed for system-message image and
  video parts to stay aligned with the real template semantics.
- On March 27, 2026, the native qwen35 CUDA lane gained the captured-graph
  greedy path, fused q/k and activation kernels, and the MMVQ-backed greedy
  output-head fast path that materially raised local greedy throughput on the
  Psionic side.
- On March 30, 2026, the repo added the first retained shared RVLLM runtime
  harvest packet at `docs/PSION_RVLLM_CUDA_GRAPH_POOL.md`, making CUDA-graph
  hits, misses, captures, shape drift, and refusal posture machine-visible
  across both native `qwen35` and native `gpt_oss` decode lanes instead of
  keeping that truth as lane-local benchmark folklore.
- The same runtime-harvest pass now also exposes one explicit cuBLAS warmup
  and handle-reuse packet at `docs/PSION_RVLLM_CUBLAS_WARMUP.md`, together
  with machine-readable `psionic_cuda_startup` startup evidence in
  `qwen35_cuda_bench --json-out`.
- The older March 27 greedy qwen35-versus-Ollama numbers on this checkout are
  now historical only. The older harness omitted explicit Ollama greedy
  settings and therefore let Ollama use its default sampler surface instead of
  a forced greedy contract.
- On March 28, 2026, after adding a native one-row CUDA top-k candidate output
  path and routing qwen35 sampled decode through `TopKCandidates { top_k }`
  instead of unconditional dense-vocab readback, and after refreshing the
  local sampler surface to honor `min_p`, `typical_p`, `mirostat`,
  `mirostat_tau`, `mirostat_eta`, and request-level `repeat_last_n`, the same
  host now has a clean committed rerun with explicit divergence evidence in
  `docs/QWEN35_OLLAMA_COMPARISON.md`.
- The same March 28, 2026 refresh widened the Psionic-side sampler and request
  surface to include `typical_p`, `mirostat`, `mirostat_tau`,
  `mirostat_eta`, and request-level `repeat_last_n`, but those controls are
  not part of the canonical Psionic-versus-Ollama matrix on this checkout
  because the local `ollamarunner` qwen3.5 path does not wire them through the
  same active sampler path.
- On March 28, 2026, after widening the shared-memory CUDA top-k fast path to
  the full bounded qwen35 envelope and then replacing the older one-row
  radix-sort route with a partitioned multi-block one-row candidate path, the
  same host now publishes the larger bounded-candidate `top_k = 100` contract
  with row-strength classification instead of summary-only throughput claims.
- Later on March 28, 2026, routing the canonical `top_k = 40` sampled
  contract through that same partitioned one-row selector at the inclusive
  threshold removed the prior sampled overhead cliff on the clean RTX 4080
  host and restored Psionic throughput leadership on all four clean
  length-matched sampled rows.
- Later the same day, tuning the partitioned one-row top-k block count from
  `8` to `24` on the same idle RTX 4080 host widened the sampled lead again.
  The follow-on rerun at
  `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_200654_archlinux-.json`
  raised Psionic `sampled_topk40` throughput from `470.49`, `243.56`,
  `175.06`, `108.36 tok/s` to `506.49`, `252.83`, `179.55`, `110.13 tok/s`,
  and raised `sampled_topk100` throughput from `446.26`, `236.17`, `171.24`,
  `106.62 tok/s` to `492.81`, `247.57`, `177.28`, `109.02 tok/s`, while the
  row-strength classifications stayed unchanged.
- Later again on March 28, 2026, current `main` made the partitioned block
  count adaptive by requested `top_k` instead of using the same fixed value
  for every bounded sampled row. The commit-pinned rerun at
  `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260328_210428_archlinux-.json`
  kept `sampled_topk40` effectively flat at `505.33`, `252.94`, `179.42`,
  `110.12 tok/s` while raising `sampled_topk100` again to `501.50`,
  `250.76`, `178.31`, `109.42 tok/s`.
- On March 29, 2026, doubling the partitioned top-k tile width by raising
  `kLogitsTopKItemsPerThread` from `8` to `16` produced another clean sampled
  gain on the same idle RTX 4080 host. The rerun at
  `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260329_003921_archlinux-.json`
  raised `sampled_topk40` to `511.51`, `254.07`, `180.01`, `110.35 tok/s`
  against Ollama `337.58`, `206.38`, `144.10`, `96.37 tok/s` while keeping
  the same bounded-candidate output mode and row-strength classifications.
- Later on March 29, 2026, the wider tile changed the best small-lane
  partitioned block shape too. Retuning the `top_k = 40` small profile from
  `24` to `48` on the same idle RTX 4080 host produced another clean sampled
  gain at
  `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260329_004540_archlinux-.json`,
  raising `sampled_topk40` again to `519.35`, `256.03`, `180.99`,
  `110.68 tok/s` against Ollama `337.91`, `206.37`, `144.43`, `96.55 tok/s`
  while keeping the same bounded-candidate output mode and row-strength
  classifications.
- Later the same day, changing the partitioned one-row top-k block shape from
  `128 threads x 16 items` to `256 threads x 8 items` produced the next full
  sampled rerun on the idle RTX 4080 host at
  `fixtures/qwen35/benchmarks/qwen35_ollama_matrix_20260329_011606_archlinux-.json`.
  That rerun kept clean `sampled_topk40` effectively flat at `518.74`,
  `256.25`, `181.13`, `110.75 tok/s` while raising `sampled_topk100` to
  `513.06`, `253.55`, `179.74`, `109.98 tok/s` against Ollama `322.06`,
  `205.61`, `144.73`, `97.59 tok/s`, with the same bounded-candidate output
  mode and row-strength classifications.
- The fresh clean-host March 28, 2026 rerun on the same RTX 4080 changes the
  canonical interpretation:
  - raw greedy `tok/s` is higher on Psionic across all four models, and
    `qwen3.5:2b` is now a `strong` exact-match row with matching EOS
    termination across all repeats
  - greedy `qwen3.5:0.8b`, `qwen3.5:4b`, and `qwen3.5:9b` still remain
    `mismatched`
  - clean sampled `top_k = 40` rows stay on the bounded candidate lane, remain
    length-matched, and now beat Ollama on all four models even though token
    divergence still starts within the first few generated tokens
  - sampled `top_k = 100` rows remain `mismatched`, but Psionic now still
    leads Ollama on all four of those rows on the clean host
- Later on March 28, 2026, zeroing the per-request hybrid SSM state on request
  init removed the old `qwen3.5:4b` cap-hit corruption on both greedy and
  `top_k = 100` sampled reruns while preserving the lead over Ollama on that
  row.
- On March 28, 2026, the same bounded qwen35 CUDA sampled lane was widened
  again to apply repeat, presence, and frequency penalties on device before
  exact top-k selection instead of forcing explicit dense `raw_logits`
  readback. On the local short-prompt smoke contract with `temperature = 0.8`,
  `top_k = 40`, `top_p = 0.9`, `min_p = 0.05`, `repeat_penalty = 1.1`,
  `repeat_last_n = 64`, `presence_penalty = 0.2`,
  `frequency_penalty = 0.1`, and `seed = 42`, the qwen35 lane stayed on
  `qwen35_output_modes=[top_k_candidates:40]` with `qwen35_raw_logits=false`
  across all four local rows and measured about `89 tok/s` on `qwen3.5:0.8b`,
  about `121 tok/s` on `qwen3.5:2b`, about `56 tok/s` on `qwen3.5:4b`, and
  about `43 tok/s` on `qwen3.5:9b`.
- The 4B row only became correct and faster after fixing the fused decode
  output head for mixed `Q4_K` and `Q6_K` weights. Greedy `ArgmaxOnly` decode
  now routes `Q6_K` output weights through `Q8_1` projection plus `argmax_f32`
  instead of falling back to the slower generic quantized matvec path.
- The 9B row also fits and runs natively on this 16 GB host. The only extra
  benchmark requirement is operational: unload Ollama's resident GPU caches
  before measuring Psionic, because Ollama keeps prior model weights live in
  VRAM.
- The qwen35 lane is materially faster than the earlier pilot on this host,
  and the current canonical matrix is now "ahead everywhere" on raw `tok/s`
  while still mixed on parity quality: greedy raw `tok/s` is higher on all
  four models with one `strong` exact-match row, clean sampled `top_k = 40`
  rows are ahead on all four models but only `weak_length_matched_only`, and
  the remaining headroom is still in greedy parity and the broader exact-match
  divergence work.
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
