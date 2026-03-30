# PSION RVLLM Attention Backend

> Status: landed on 2026-03-30 as the eighth retained RVLLM runtime-harvest
> packet.

This document records the current attention-backend selection seam for the
admitted CUDA serving lane.

The important truth here is not that Psionic needed new CUDA attention
kernels. It did not. The repo already shipped multiple real attention decode
backends. The gap was that kernel-family selection was still encoded as nested
callsite branching instead of one explicit selector contract.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_attention_backend_v1.json`

Current retained truth:

* packet digest `eca9f77ad8551b44a4175d05f450777accb3d218672f75a766f6298a3d20ac72`
* default backend:
  - `dense_f16_kv`
* alternate backends:
  - `dense_f16_kv_q8_1_output_fusion`
  - `turboquant_kv`
* selection inputs:
  - `use_turboquant_kv`
  - `use_q8_1_attention_output_fusion`
  - `use_graph_attention`
* capability gates:
  - `cuda_kv_cache_encoding_selection`
  - `q8_1_attention_output_fusion_capability`
  - `graph_replay_admission`

Retained validation surface:

* `selector_defaults_to_dense_f16_kv_backend`
* `selector_uses_q8_1_output_fusion_backend_when_enabled`
* `selector_prefers_turboquant_backend`
* `cuda_kv_cache_encoding_selection_activates_turboquant_when_supported`

Claim boundary:

* This packet does **not** claim a backend swap happened.
* It does claim the current attention-backend seam is now explicit: dense f16
  KV remains the default, alternates stay capability-gated, and later backend
  experiments no longer require route logic to be rewritten at every callsite.
