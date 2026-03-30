# PSION RVLLM cuBLAS Warmup

> Status: landed on 2026-03-30 as the second retained RVLLM runtime-harvest
> packet.

This document records the first explicit cuBLAS handle-reuse and warmup packet
for the admitted native CUDA serving lanes.

The current CUDA backend already owned real handle creation and stream binding.
What was missing was one machine-readable startup contract that says warmup is
intentional, outside the user-billed request path, and visible in startup
evidence.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_cublas_warmup_v1.json`

Current retained truth:

* packet digest `ab8ec999b8e80605744171d3adab152b6ffaecda22877b321ec4c65bb084ea36`
* runtime scope:
  - `cuda_backend_runtime`
  - `qwen35.native_cuda_decode`
  - `gpt_oss.native_cuda_decode`
* cuBLAS handle scope `per_device_runtime_owner`
* cuBLAS stream binding `bind_stream_per_submission`
* warmup stage `explicit_admitted_startup_request`
* startup report fields:
  - `warmup_status`
  - `prompt_latency_ns`
  - `decode_latency_ns`
  - `total_latency_ns`
  - `output_tokens`
  - `request_billed_to_user`
* retained qwen35 startup report:
  - status `explicit_warmup_completed`
  - prompt latency `11200000 ns`
  - decode latency `12600000 ns`
  - total latency `23800000 ns`
  - output tokens `8`
  - request billed to user `false`

Operator surface:

* `qwen35_cuda_bench --json-out` now carries `psionic_cuda_startup`
* the startup report makes explicit:
  - the cuBLAS handle is reused by runtime owner
  - stream binding remains explicit per submission
  - warmup completed before measured user-facing runs

Claim boundary:

* This is runtime startup-path truth, not training-recipe warmup.
* It does not claim universal cold-start elimination.
* It does make the first-step stabilization path explicit and machine-visible.
