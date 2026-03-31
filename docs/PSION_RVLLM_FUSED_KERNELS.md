# PSION RVLLM Fused Kernels

> Status: landed on 2026-03-30 as the tenth retained RVLLM runtime-harvest
> packet.

This document records the admitted selective fused-kernel posture for native
`qwen35` and native `gpt_oss`.

The important truth here is not that Psionic now owns a general PTX compiler.
It does not. The gap was that the repo already owned several high-value fused
CUDA kernels, but the hot-path shortlist, its feature gates, and its disable
paths were still scattered across backend tests and model-specific branches.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_fused_kernels_v1.json`

Current retained truth:

* packet digest `8e987c2a4ccf2fba8cfac6e0618a3d0e3a36ae6ec4e7ddba38eb564fbaa33f6a`
* admitted families:
  - `qwen35_qkv_rms_norm`
  - `gpt_oss_selected4_moe`
* explicit gates:
  - `PSIONIC_QWEN35_DISABLE_FUSED_QKV_RMS_NORM`
  - `PSIONIC_GPT_OSS_EXPERIMENTAL_FUSED_SELECTED4_MOE_DOWN`

Retained benchmark comparison:

| Family | Before op ms | After op ms | Before E2E ms | After E2E ms | Before tok/s | After tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen35_qkv_rms_norm` | `1.92` | `1.31` | `22.4` | `20.1` | `55.7` | `61.8` |
| `gpt_oss_selected4_moe` | `4.84` | `3.62` | `37.5` | `34.1` | `41.8` | `46.2` |

Disable-path posture:

* `qwen35_qkv_rms_norm` falls back to:
  - `split_interleaved_query_gate_f32`
  - `rms_norm` / `rms_norm_region`
  - `copy_buffer_region`
* `gpt_oss_selected4_moe` falls back to:
  - `moe_gate_up_swiglu_q8_1`
  - `expert_matvec_q8_1_ids` or `moe_down_aggregate_q8_1`

Claim boundary:

* This packet claims only the profiled admitted fused-kernel shortlist.
* It does **not** claim a general PTX emission lane, broad kernel
  auto-generation, or permission to widen fused-kernel ownership beyond the
  already-shipped hot paths.
