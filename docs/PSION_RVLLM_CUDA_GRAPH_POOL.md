# PSION RVLLM CUDA Graph Pool

> Status: landed on 2026-03-30 as the first retained RVLLM runtime-harvest
> packet in the current Psionic runtime.

This document records the first explicit shared CUDA-graph replay contract
across the admitted native CUDA decode lanes.

The runtime already had real lane-local graph capture and replay wins in
`qwen35` and `gpt_oss`. What was missing was one retained packet that makes the
graph pool contract machine-visible instead of leaving it as lane-local
folklore.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_cuda_graph_pool_v1.json`

Current retained truth:

* packet digest `47864743915a631327933f5ec2ca3d1d7bed30b426592c7aad9b84f23276e509`
* runtime scope:
  - `qwen35.native_cuda_decode`
  - `gpt_oss.native_cuda_decode`
* graph key fields:
  - `served_artifact_id`
  - `batch_size`
  - `sequence_length`
  - `decode_mode`
  - `kv_cache_encoding`
  - `cache_allocation_identity`
* admitted decode modes:
  - `argmax_only`
  - `top_k_candidates { top_k = 40 }`
  - `raw_logits`
* qwen35 graph replay evidence:
  - steps `3`
  - hits `1`
  - misses `2`
  - captures `2`
  - shape drifts `1`
  - capture latency `126000 ns`
* gpt-oss graph replay evidence:
  - steps `3`
  - hits `1`
  - misses `1`
  - captures `1`
  - refusals `1`
  - capture latency `54000 ns`

What this packet means:

* Psionic now exposes one explicit graph-replay contract instead of implying
  graph reuse from throughput anecdotes.
* Graph replay is keyed by stable request and allocation identity, not vague
  “warm request” language.
* Capture refusal, shape drift, and uncaptured fallback remain visible.
* This does **not** swap runtimes or claim blanket graph stability outside the
  admitted decode envelopes.

Operator surface:

* `crates/psionic-serve/examples/qwen35_cuda_bench.rs` now publishes:
  - `qwen35_graph_hits`
  - `qwen35_graph_misses`
  - `qwen35_graph_captures`
  - `qwen35_graph_shape_drifts`
* native qwen35 request metrics now attach `graph_replay`
* native gpt-oss performance metrics now attach `cuda_graph_replay`

Claim boundary:

* The admitted fast path is still bounded native CUDA decode in the current
  Psionic runtime.
* The graph pool packet records runtime harvest from already-shipped lanes.
* It does not claim a new scheduler, a new runtime owner, or universal graph
  reuse outside the admitted request shapes.
