# PSION RVLLM Prefill-Decode Scheduler

> Status: landed on 2026-03-30 as the seventh retained RVLLM runtime-harvest
> packet.

This document records the admitted prefill-versus-decode scheduler split for
the current Psionic serving lane.

The important truth here is not that Psionic suddenly gained split scheduling.
It already had an admitted continuous-batch policy with separate prompt-prefill
and decode token budgets, explicit realized scheduling classes, TTFT and ITL
metrics, and response headers exposing that split to operators and downstream
consumers. The gap was that the repo did not yet retain one explicit packet
binding those facts together as the current scheduler contract.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_prefill_decode_scheduler_v1.json`

Current retained truth:

* packet digest `ee4f42da62ec05ee6fe9e82e9355e3845de872cfd1b0c8950e5ef2113a39e3a1`
* admitted scheduler policy:
  - `max_active_requests = 4`
  - `max_queued_requests = 32`
  - `max_prefill_tokens_per_tick = 4`
  - `max_decode_tokens_per_tick = 8`
* realized scheduling classes:
  - `prefill`
  - `decode`
  - `mixed_prefill_decode`
  - `fallback_single_request`
* admitted execution mode:
  - `disaggregated_colocated`
* admitted handoff transport:
  - `in_process_kv_state`
* request receipt fields:
  - `queue_depth_at_admission`
  - `max_batch_size_observed`
  - `scheduling_class`
  - `prefill_tokens`
  - `decode_tokens`
  - `prefill_decode_mode`
  - `prefill_decode_handoff`
  - `time_to_first_token_ns`
  - `inter_token_latency_ns`
  - `fallback_reason`
* response headers:
  - `x-psionic-batch-posture`
  - `x-psionic-scheduling-class`
  - `x-psionic-prefill-decode-mode`
  - `x-psionic-ttft-ns`
  - `x-psionic-itl-ns`

Retained validation surface:

* `cpu_reference_continuous_batch_scheduler_mixes_prefill_and_decode`
* `generic_server_grammar_fallback_is_machine_checkable`
* `generic_server_json_schema_fallback_is_machine_checkable`

Claim boundary:

* This packet does **not** claim a second scheduler runtime or broader
  continuous-batching capability than the admitted lane currently exposes.
* It does claim that Psionic already has a real explicit prefill-versus-decode
  scheduler split with distinct TTFT and ITL truth, and that those facts are
  now retained as one machine-readable contract.
