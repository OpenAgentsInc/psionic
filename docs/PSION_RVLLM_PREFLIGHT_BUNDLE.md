# PSION RVLLM Pre-Flight Bundle

> Status: updated on 2026-03-31 to include the retained `cublasLt` plan-cache
> packet.

This document records the admitted GPU-serving pre-flight bundle for native
`qwen35` and native `gpt_oss`.

The important truth here is not that Psionic needed a second startup system.
It did not. The gap was that warmup facts were scattered across graph capture,
cuBLAS bringup, benchmark harness startup fields, and backend runtime-resource
reports. This packet binds those facts into one explicit pre-flight contract.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_preflight_bundle_v1.json`

Current retained truth:

* packet digest `a17262a58a4a9e84600e424f1ec37556d9fafe2158a34aba9ff2511604044e91`
* admitted serving paths:
  - `qwen35.native_cuda_decode`
  - `gpt_oss.native_cuda_decode`
* pre-flight steps:
  - runtime owner ready
  - allocator pool primed
  - bounded `cublasLt` plan-cache ready
  - explicit cuBLAS warmup request
  - CUDA graph capture or refusal
  - kernel-cache posture exported
* startup report fields:
  - `cublas_handle_scope`
  - `cublas_stream_binding`
  - `cublas_lt_tuning_status`
  - `cublas_lt_plan_cache_scope`
  - `cublas_lt_selected_plan_count`
  - `cublas_lt_tuned_shape_count`
  - `cublas_lt_fallback_shape_count`
  - `cublas_lt_max_workspace_bytes`
  - `cublas_lt_selected_plans`
  - `warmup_status`
  - `warmup_prompt_s`
  - `warmup_decode_s`
  - `warmup_total_s`
  - `warmup_output_tokens`
  - `request_billed_to_user`
* allocator-pool posture:
  - `policy = exact_tensor_spec`
  - `max_cached_buffers = 128`
  - `max_cached_bytes = 67108864`
* kernel-cache posture:
  - exported as machine-visible runtime evidence
  - remains disabled on the admitted lane unless explicitly enabled later
* referenced retained packets now include:
  - CUDA graph pool
  - `cublasLt` plan cache
  - cuBLAS warmup

Claim boundary:

* This packet does **not** claim all cold cost disappears.
* It does claim the admitted runtime now records cold versus warm posture in
  one machine-readable bundle instead of hiding it inside the first user
  request.
* Warmup success, `cublasLt` tuning status, and fallback counts remain explicit.
