# PSION-RVLLM cuBLASLt Plan Cache

This packet pins the admitted `cublasLt` GEMM-plan selection lane that now
exists inside native CUDA decode for `qwen35` and `gpt_oss`.

## Scope

- runtime: `cuda_backend_runtime`
- model families:
  - `qwen35.native_cuda_decode`
  - `gpt_oss.native_cuda_decode`
- admitted backend route: `cublaslt_f16_to_f32`
- plan-cache scope: `per_weight_scope_shape_dtype_route`

## Admitted Shapes

- input dtype: `f16`
- output dtype: `f32`
- bounded startup row ladder: `1`, `8`, `32`
- maximum admitted rows: `32`

The runtime tunes only the representative decode shapes that matter for the
served lane. Everything outside that admitted posture falls back explicitly to
the existing cuBLAS GEMM path instead of pretending the tuned path is universal.

## Representative Scopes

- `qwen35.native_cuda_decode/output_logits`
- `qwen35.native_cuda_decode/ffn_gate_up`
- `qwen35.native_cuda_decode/ffn_down`
- `qwen35.native_cuda_decode/hybrid_qkv_gate_alpha_beta`
- `qwen35.native_cuda_decode/hybrid_ssm_out`
- `qwen35.native_cuda_decode/attention_qkv`
- `qwen35.native_cuda_decode/attention_output`
- `gpt_oss.native_cuda_decode/output_logits`
- `gpt_oss.native_cuda_decode/attention_qkv`
- `gpt_oss.native_cuda_decode/attention_output`

## Required Evidence

- `psionic_cuda_startup.cublas_lt_tuning_status`
- `psionic_cuda_startup.cublas_lt_plan_cache_scope`
- `psionic_cuda_startup.cublas_lt_selected_plan_count`
- `psionic_cuda_startup.cublas_lt_tuned_shape_count`
- `psionic_cuda_startup.cublas_lt_fallback_shape_count`
- `psionic_cuda_startup.cublas_lt_max_workspace_bytes`
- `psionic_cuda_startup.cublas_lt_selected_plans[].model_family`
- `psionic_cuda_startup.cublas_lt_selected_plans[].op_kind`
- `psionic_cuda_startup.cublas_lt_selected_plans[].rows`
- `psionic_cuda_startup.cublas_lt_selected_plans[].inner`
- `psionic_cuda_startup.cublas_lt_selected_plans[].cols`
- `psionic_cuda_startup.cublas_lt_selected_plans[].backend_route`
- `psionic_cuda_startup.cublas_lt_selected_plans[].workspace_bytes`
- `psionic_cuda_startup.cublas_lt_selected_plans[].mean_time_us`
- `psionic_cuda_startup.cublas_lt_selected_plans[].algorithm_fingerprint`

## Validation

- `cargo test -p psionic-serve psion_rvllm_cublaslt_plan_cache::tests::builtin_packet_matches_committed_fixture --lib -- --exact`
- `cargo build -p psionic-serve --example qwen35_cuda_bench`
- `cargo build -p psionic-serve --bin psionic-openai-server`

## Current Truth

- packet digest: `9c1c66447888d4e7c91c3220cbf730ea87ead1b14ad66997d4919db40fce730b`
- fixture: `fixtures/psion/serve/psion_rvllm_cublaslt_plan_cache_v1.json`

## Claim Boundary

This packet does not claim that every decode GEMM now uses `cublasLt`. It
claims that the admitted native CUDA serving lane has one explicit bounded
startup autotune surface, one explicit plan-cache key, and one explicit startup
receipt that says which representative shapes were tuned and which shapes stayed
on fallback.
