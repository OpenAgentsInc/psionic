# Inference Engine

Psionic is only inference-ready when it can honestly serve compute products on
admitted hardware classes rather than just run tensor math.

## Hardware-First Rule

Psionic does not start by picking one upstream engine brand and then forcing
every lane through that shape.

It starts by fixing:

- hardware class and backend family
- residency and memory posture
- single-node versus clustered topology
- admitted serving role
- capability, refusal, and latency publication

The runtime lane follows from that decision.

`docs/HARDWARE_VALIDATION_MATRIX.md` owns the minimum shipped hardware rows and
their admitted roles. This document owns the serving-runtime contract above
those rows.

If two lanes share one API surface but differ in hardware class, topology, or
admitted role, they must publish different runtime truth instead of collapsing
into one generic engine claim.

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
- `Gemma 4` now has one published bounded Psionic lane on this checkout.
  The canonical publication doc is
  `docs/NON_GPT_OSS_GEMMA4_PILOT.md`.
  The first frozen target remains:
  - artifact = `gemma4:e4b`
  - family shape = dense `Gemma 4`
  - first claim = one bounded text-generation lane on CUDA
  - first publication bar = truthful backend, execution mode, execution
    engine, and refusal metadata on the generic OpenAI-compatible server
- The published bounded `Gemma 4` lane now provides:
  - native Psionic CUDA runtime admission for dense quantized GGUF projection
    artifacts
  - `backend = cuda`
  - `execution_mode = native`
  - `execution_engine = psionic`
  - `/v1/chat/completions` and `/v1/responses` on the generic
    OpenAI-compatible server
  - typed promoted-revision adoption for `gemma4:e4b` from the bounded Gemma
    trainer, with checkpoint-plus-export revalidation, active served-revision
    provenance, fail-closed mismatch refusal, and rollback to the last
    known-good revision
  - Gemma-native tool calling through explicit
    `<|tool_call>call:<tool>{...}<tool_call|>` blocks with JSON-schema-subset
    argument validation
  - replayable `/v1/responses` state across assistant tool turns and replayed
    tool results
  - published `multimodal_projection_mode = processor_owned` for `image` and
    `video`, with explicit refusal on the current generic OpenAI surface until
    a real Gemma processor lands
  - published `audio_input_mode = processor_owned` with
    `audio_input_parts = ["input_audio"]` on dense `e2b` and `e4b` rows only,
    with explicit refusal on the current generic OpenAI surface until a real
    Gemma audio processor lands
  - bounded prompt-render, server-smoke, refusal, and repeatable CUDA
    conformance coverage for `gemma4:e4b`
- `Gemma 4` also now has one second dense validation lane outside the first
  published claim:
  - artifact class = dense `31B`
  - operator hook = `PSIONIC_GEMMA4_31B_PILOT_GGUF_PATH`
  - validation scope = the same checked-in prompt fixture, CUDA backend truth,
    `/v1/chat/completions` and `/v1/responses` endpoint set, structured-output
    refusal posture, and processor-owned image/video refusal posture as the
    `e4b` lane
  - claim boundary = validation only; it does not widen the original
    `gemma4:e4b` publication
- `Gemma 4` now also has one first sparse-family publication contract outside
  the dense first claim:
  - artifact class = sparse `gemma4:26b`
  - runtime contract = `family_specific_placement`
  - published topology = the sparse expert topology inspected from the loaded
    GGUF; current retained cluster fixtures still exercise the older
    `expert_count = 64`, `active_expert_count = 4`,
    `expert_feed_forward_length = 4096` shape, so topology-value reconciliation
    remains a follow-on consistency task instead of part of this lane claim
  - publication surfaces = `/health`, `/v1/models`, routed inventory, and mesh
    management status all carry the same sparse-topology truth
  - default request posture = the local generic OpenAI-compatible server now
    admits one single-node text lane for `gemma4:26b` on `cuda` and `metal`
    without requiring a distributed sparse schedule
  - single-node hardware posture = one host must admit the full quantized
    sparse GGUF plus active KV cache on the chosen backend; otherwise the
    server must fail closed or the operator must admit a distributed sparse
    schedule instead
  - admitted request posture = once the operator admits a real
    two-or-more-node sparse schedule, the same server can execute
    `gemma4:26b` and publish `disposition = sharded`,
    `execution_topology = tensor_sharded`, and request-specific expert-routing
    proof in response headers and `psionic_cluster_execution`
  - shard lifecycle = admitting that sparse schedule now also materializes and
    caches one explicit shard artifact per expert-host assignment, publishes
    shard build cache keys and artifact digests into routed inventory, `/health`,
    `/v1/models`, and mesh management status, and reuses the same shard
    artifacts on repeated admission instead of silently pretending every sparse
    turn is cold
  - stateful locality = `/v1/responses` now persists one sparse placement
    binding per conversation so follow-up turns stay on the same worker and
    placement digest while that shard state remains healthy, then rebind
    cleanly if the placement changes
  - current Metal single-node closure = the native local Metal path now keeps
    all `30 / 30` FFN layers on device for the real
    `gemma-4-26B-A4B-it-Q4_K_M.gguf` artifact, including fused `Q4_K` sparse
    gate/up experts and `Q5_0` dense plus sparse down projections; the first
    retained `2026-04-14` benchmark receipt moved from about `5.14 tok/s` to
    about `24.22 tok/s`, and the follow-on device-resident decode pass moved
    the same retained prompt again to about `29.58 tok/s` while keeping greedy
    readback bounded to `4 B/token` and timed-request host KV materialization
    at `0`
  - current competitive benchmark gate = the repo now also has one fail-closed
    same-host benchmark gate for this exact lane at
    `scripts/release/run-gemma4-26b-competitive-benchmark.sh`; the retained
    `2026-04-15` receipt keeps the sparse path honest with
    `native_sparse_execution = true`,
    `host_fallback_observed = false`,
    `sparse_ffn_backend = metal_grouped_experts`, and
    `router_backend = metal_router_topk_softmax`, but still reports
    `decode_tok_s = 30.63` for Psionic versus `102.34` on `ollama` and
    `113.79` on `llama.cpp`, so the gate currently fails for competitive
    throughput rather than sparse-receipt dishonesty; the canonical audit is
    `docs/audits/2026-04-15-gemma4-26b-competitive-benchmark-gate-audit.md`
  - current boundary = that pass fixes the local fallback cliff but does not
    yet close parity with `ollama` or `llama.cpp`, and the sparse 26B local
    lane still returns malformed text on the shared benchmark prompt
  - current follow-on tuning note = a same-day retained branch pass widened
    the dense `q5_0` / `q8_0` Metal matvec threadgroups and restored
    dense-`f32` KV rows as the default Metal cache policy while keeping the
    real `DenseF16Mirror` path behind
    `PSIONIC_METAL_KV_CACHE_F16_MIRROR=1`; the latest retained local check on
    the same benchmark prompt still lands at about `30.20 tok/s`, so the lane
    remains materially behind `ollama` and `llama.cpp`
  - claim boundary = one local single-node text lane plus one admitted
    distributed sparse extension only; this still does not promote multimodal,
    audio, structured-output, or training claims for `gemma4:26b`
- The first real distributed `Gemma 4` proof is now one split
  `pipeline_sharded` `gemma4:e4b` request across two CUDA machines:
  - execution target = dense `gemma4:e4b`
  - realized topology = `pipeline_sharded`
  - realized disposition = `sharded`
  - Metal prefix execution now keeps the split-stage layer stack on the device
    and only materializes the handoff hidden state and forwarded KV rows that
    the remote suffix actually needs
  - request publication now exposes the clustered path directly in response
    headers and response or receipt provenance instead of inferring it from a
    remote route
- The generic OpenAI-compatible server, routed inventory, and mesh management
  status now also publish one family-agnostic clustered-execution summary for
  every admitted model row:
  - generic execution modes = `remote_whole_request`, `replicated`,
    `dense_split`, `sparse_expert`
  - generic topology kinds = `replicated`, `pipeline_sharded`,
    `layer_sharded`, `tensor_sharded`
  - publication surfaces = `/health`, `/v1/models`, routed inventory, and
    mesh management status
  - local routed detail = routed inventory and mesh management status can also
    carry one runtime-owned cluster capability profile that explains which
    multi-machine path is actually admitted on that row
  - intent = downstream product surfaces no longer need `gpt_oss`-specific
    inference to tell whether a model is proxied to one remote host, replicated
    across hosts, split across hosts, or running as a sparse expert topology
- The older bootstrap path still exists, but it remains explicitly classified
  as remote whole-request proxying:
  - remote execution target = CUDA-backed `gemma4:e4b`
  - routed publication keeps remote route truth explicit on `/v1/models` with
    `route_backend = cuda`, `route_execution_mode = native`, and
    `route_execution_engine = psionic`
  - thin-client served truth stays separate and honest with
    `served_backend = remote` and `execution_mode = proxy`
  - routed remote publication now keeps the same admitted endpoint set as the
    local dense lane instead of silently narrowing back to chat-only
- `psionic-runtime` now also owns one bounded dense multi-device execution core
  below the served product surface:
  - admitted topology = ordered `pipeline_sharded` or `layer_sharded`
    execution over explicit layer-range shard artifacts
  - execution posture = adjacent shard runtimes prefer direct worker-to-worker
    stage handoff when both sides support it and otherwise fall back to the
    host-mediated path, with final output assembly on the last stage
  - `gemma4:e4b` now has one bounded proof on top of that runtime-core path
    through runtime reports, provider receipts, and generic-server response
    headers
  - claim boundary = one bounded split dense path only; this does not by
    itself promote broader wallet-settled or sparse-family served product
    claims
- `Gemma 4` now also has one first-class Metal lane contract on the generic
  OpenAI-compatible server:
  - `backend = metal`
  - `execution_mode = native`
  - `execution_engine = psionic`
  - `residency_mode = metal_accelerated`
  - `fallback_policy = refuse`
  - the same `/v1/chat/completions` and `/v1/responses` publication surface as
    the admitted CUDA lane
  - no scheduler policy claim
  - current execution posture = live native single-node execution on Apple
    Silicon for the admitted local Gemma text lanes, including dense
    `gemma4:e4b` and sparse `gemma4:26b`
- The first bounded `Gemma 4` claim must stay explicit about its unsupported
  regions:
  - admitted image execution on the processor-owned lane
  - admitted video execution on the processor-owned lane
  - admitted audio execution on the processor-owned audio lane
  - `31B`
  - broader `gemma4:26b` support beyond the admitted single-node text lane and
    optional admitted distributed sparse extension
  - full Metal parity across Gemma artifacts and clustered roles
  - generic structured outputs
  - unquantized projection tensors on the native CUDA lane
  - full parity with `llama.cpp` or `ollama`
- That unsupported-region list is the boundary of the first published
  `gemma4:e4b` claim. The optional dense `31B` validation lane keeps the same
  refusal posture and publication shape, but it still does not promote `31B`
  into the published first claim automatically.
- The dense `Gemma 4` audio lane is narrower than the family-wide multimodal
  publication:
  - `e2b` and `e4b` publish `audio_input_mode = processor_owned` and
    `audio_input_parts = ["input_audio"]`
  - `31B` and sparse `gemma4:26b` do not publish audio capability and fail
    closed for direct `input_audio` parts on the current generic surface
- CPU-only debug bring-up may still be useful while the lane is under active
  development. The repo now admits `Gemma 4` on CPU for bounded debug
  execution, but CPU still does not satisfy the first published `Gemma 4`
  support claim for Psionic.
- `qwen35` is `implemented_early` through a native Psionic CUDA text-generation
  runtime with prompt-projected image and video inputs at the HTTP layer.
- Current `main` also admits native local execution for the real Hugging Face
  `Qwen3.5-27B-Q4_K_M.gguf` artifact without a llama.cpp proxy on both CPU and
  Metal. That path now accepts the scalar `qwen35.attention.head_count_kv`
  form, the `blk.N.ssm_dt.bias` tensor naming used by the public 27B artifact,
  the official `qwen35` tokenizer pre, GGUF MRoPE metadata, and the missing
  `qwen35.ssm.v_head_reordered` family fact by defaulting it to `true`.
- The current native Metal `qwen35` lane is coherent and materially faster than
  the first bring-up, but it is still host-stepped and not yet in the same
  performance class as the local Ollama / llama.cpp reference. The current gap
  is execution shape, not GGUF admission.
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
  through `--json-out`. The native direct row now also keeps
  `benchmark_class`, `load_s`, per-run `ttft_s`, per-run `itl_s`, and mean
  TTFT / ITL fields explicit in the direct receipt instead of leaving those
  timings trapped inside the runtime metrics lane.
- The repo-owned direct-versus-HTTP collector for the admitted native qwen35
  CUDA lane now lives at `scripts/release/qwen35_direct_vs_http_compare.py`.
  It runs the native direct-engine receipt and the native
  `psionic-openai-server` receipt on one explicit prompt contract, keeps the
  two benchmark classes separate in the published JSON, records HTTP startup
  and warmup timing, writes one explicit concurrency ladder instead of
  folding runtime and server overhead into one number, and now requires the
  native direct row to pass the fallback-free CUDA publication gate unless an
  operator opts into compatibility receipts explicitly.
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
- On March 31, 2026, the generic OpenAI-compatible `/v1/chat/completions`
  stream path stopped collapsing plain-text responses into one final SSE delta
  and now emits incremental content chunks instead. The same pass also stopped
  rejecting auto tool-mode turns when the model returned plain assistant text
  without a machine-readable tool envelope.
- That stream improvement is still bounded on native `qwen35`: the server now
  emits incremental SSE content deltas, but the native `qwen35` backend does
  not yet publish true decoder-time token events. The current lane still
  materializes the response before chunk emission at the server surface.
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
- On March 31, 2026, the same runtime-harvest line gained one explicit
  `cublasLt` plan-cache packet at `docs/PSION_RVLLM_CUBLASLT_PLAN_CACHE.md`,
  making admitted startup autotune status, selected-plan receipts, fallback
  shape counts, and workspace posture machine-visible for native `qwen35` and
  native `gpt_oss` CUDA decode.
- Later on March 31, 2026, the admitted native `qwen35` CUDA decode lane
  moved the device token-embedding mirror, request decode params, and
  initial-token seed inside the request-local captured graph for the argmax,
  top-k, and raw-logit output branches, and the repo added captured-lane T>1
  parity tests for those same branches against the debug-attention reference
  path in `crates/psionic-serve/src/openai_http.rs`.
- The same issue-805 pass also published one explicit before/after receipt
  pair at
  `fixtures/qwen35/benchmarks/qwen35_cuda_issue_805_20260331_archlinux_nongated.json`
  and `fixtures/qwen35/benchmarks/qwen35_cuda_issue_805_20260331_archlinux.json`.
  The compatibility receipt measured about `492.08 tok/s` with one graph shape
  drift per request; the admitted rerun measured about `502.39 tok/s` with
  `qwen35_graph_hits=27`, one initial graph capture plus one matching miss,
  `qwen35_graph_shape_drifts=0`, `qwen35_readback_bytes=224`, zero host
  fallback evidence, and
  `qwen35_attention_backends=[fa3_split_kv_f16_kv_graph@split1]`.
- The same pass now also retains one explicit GPU logits-selection packet at
  `docs/PSION_RVLLM_GPU_LOGITS_SELECTION.md`, binding the already-shipped
  qwen35 and gpt-oss device-argmax / bounded-candidate lanes to explicit
  readback-byte and raw-logits fallback truth.
- The same pass now also retains one explicit sampling-loop packet at
  `docs/PSION_RVLLM_SAMPLING_LOOP.md`, making the admitted seeded sampler,
  exact bounded-candidate replay lane, and sparse penalty scratch-reuse path
  machine-visible instead of leaving that hot-path truth buried inside
  `qwen35.rs` and `gpt_oss.rs`.
- The same pass now also retains one explicit pre-flight bundle packet at
  `docs/PSION_RVLLM_PREFLIGHT_BUNDLE.md`, tying together graph capture,
  cuBLAS warmup, allocator-pool posture, kernel-cache posture, and cold-versus-
  warm startup evidence for the admitted CUDA serving lane.
- The same pass now also retains one explicit fallback-free CUDA benchmark
  gate packet at `docs/PSION_RVLLM_FALLBACK_FREE_CUDA_GATE.md`, making host
  fallback evidence, raw-logit materialization, graph-stability checks, and
  refusal posture machine-visible for the admitted native qwen35 greedy lane
  instead of silently publishing degraded direct-engine rows.
- The same pass now also retains one explicit paged-KV manager packet at
  `docs/PSION_RVLLM_PAGED_KV_MANAGER.md`, making logical page layout,
  owner-bound growth accounting, spill policy, residency movement, and refusal
  posture explicit instead of leaving the block-manager contract implicit.
- The same pass now also retains one explicit prefill/decode scheduler packet
  at `docs/PSION_RVLLM_PREFILL_DECODE_SCHEDULER.md`, making the admitted
  continuous-batch policy, realized scheduling classes, TTFT/ITL exposure, and
  response-header contract machine-visible instead of leaving that split buried
  across runtime receipts and HTTP glue.
- The same pass now also retains one explicit attention-backend packet at
  `docs/PSION_RVLLM_ATTENTION_BACKEND.md`, making the current CUDA attention
  backend selector explicit so dense f16 KV remains the default while
  turboquant KV and q8_1 output fusion stay capability-gated alternates.
- The same pass now also retains one explicit FA3-class decode-attention
  packet at `docs/PSION_RVLLM_FA3_DECODE_ATTENTION_BACKEND.md`, making the
  admitted qwen35 CUDA graph backend, split-KV heuristic, SM gate, and
  downgrade evidence machine-visible instead of leaving the graph decode kernel
  identity implicit.
- The same pass now also retains one explicit memory-pool packet at
  `docs/PSION_RVLLM_MEMORY_POOL.md`, making the exact-spec CUDA allocator pool
  real for the admitted decode lane and binding before/after allocation count,
  steady-state memory, and long-run leak posture into one explicit runtime
  packet instead of leaving allocator reuse as an unimplemented contract.
- The same pass now also retains one explicit fused-kernel shortlist packet at
  `docs/PSION_RVLLM_FUSED_KERNELS.md`, making the admitted qwen35 QKV/RMSNorm
  and gpt-oss selected4 MoE kernels explicit together with their env-gated
  disable paths instead of leaving fused-kernel ownership as broad folklore.
- The same pass now also retains one explicit KV eviction-and-reuse packet at
  `docs/PSION_RVLLM_KV_EVICTION_REUSE.md`, making oldest-page eviction,
  reclaimed-page reuse, predictive reuse reporting, and long-context stress
  truth explicit instead of leaving that policy as implicit page churn.
- The same pass now also retains one explicit direct-engine comparator packet
  at `docs/PSION_RVLLM_DIRECT_ENGINE_COMPARATOR.md`, making the admitted
  native direct row, native HTTP row, and optional direct `vllm` reference row
  explicit as separate benchmark classes instead of leaving runtime-versus-
  server attribution to ad hoc local notes.
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
