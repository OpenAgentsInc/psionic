# PSION-RVLLM FA3 Decode Attention Backend

This packet pins the admitted FA3-class decode-attention lane that now exists
inside native qwen35 CUDA graph decode.

## Scope

- runtime: `qwen35.native_cuda_decode`
- graph lane: `qwen35.greedy_cuda_graph_decode`
- kernel owner: `psionic_backend_cuda.attention_decode_kernels`

## Backend Contract

- requested backend: `fa3_split_kv_f16_kv_graph`
- graph fallback backend: `dense_f16_kv_graph_legacy`
- non-graph backend: `dense_f16_kv_legacy`
- architecture gate: `cuda_compute_capability>=8.0`
- shape gates:
  - `head_count % kv_head_count == 0`
  - `heads_per_group <= 8`
  - `head_dim <= 256`

## Split Heuristic

- `ctx <= 512`: `1` split
- `ctx <= 2048`: `2` splits
- `ctx <= 8192`: `4` splits
- `ctx > 8192`: `8` splits

The runtime publishes the executed split count per backend receipt. If the
graph lane cannot admit the FA3 backend, the receipt keeps the downgrade
explicit instead of silently reusing the legacy graph kernel.

## Required Evidence

- `metrics.qwen35_cuda_decode.attention_backend.layer_invocation_count`
- `metrics.qwen35_cuda_decode.attention_backend.executions[].requested_backend`
- `metrics.qwen35_cuda_decode.attention_backend.executions[].executed_backend`
- `metrics.qwen35_cuda_decode.attention_backend.executions[].fallback_reason`
- `metrics.qwen35_cuda_decode.attention_backend.executions[].split_count`
- `metrics.qwen35_cuda_decode.attention_backend.executions[].compute_capability`
- `bench.runs[].qwen35_attention_backends`

## Validation

- `cargo test -p psionic-serve psion_rvllm_fa3_decode_attention_backend::tests::builtin_packet_matches_committed_fixture --lib -- --exact`
- `cargo test -p psionic-backend-cuda cuda_submission_fused_attention_graph_fa3_f16_kv_matches_legacy_reference_when_available --lib -- --exact`
- `cargo build -p psionic-serve --example qwen35_cuda_bench`
- `cargo build -p psionic-serve --bin psionic-openai-server`

This checkout adds the runtime surface and parity guard. Real throughput deltas
still have to be measured on an idle CUDA host before Psionic can publish a
performance claim for the admitted lane.
