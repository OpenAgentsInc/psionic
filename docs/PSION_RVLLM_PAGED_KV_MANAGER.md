# PSION RVLLM Paged-KV Manager

> Status: landed on 2026-03-30 as the sixth retained RVLLM runtime-harvest
> packet.

This document records the current block-manager and paged-KV truth beneath the
admitted Psionic serving lane.

The important truth here is not that Psionic lacked paged KV. It already had
explicit `KvCachePolicy`, `KvCacheState`, owner-bound accounting, spill
posture, and host-device residency movement. The gap was that the repo did not
yet retain one explicit packet tying those surfaces together as the current
block-manager contract.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_paged_kv_manager_v1.json`

Current retained truth:

* packet digest `334e915310db56529dab6ccf0a3687caf6d865e9836a103b965a995b223719c5`
* runtime scope:
  - `psionic_runtime.kv_cache_policy`
  - `psionic_serve.inmemory_kv_cache`
  - `gpt_oss.cuda_kv_cache`
* default logical page size:
  - `16` tokens per page on the reference-path default policy
* owner classes:
  - `request`
  - `session`
  - `shared_prefix`
* spill policies:
  - `refuse_new_pages`
  - `evict_oldest_pages`
  - `spill_to_host`
* residency tiers:
  - `device`
  - `host`
  - `distributed`
* residency movement kinds:
  - `prefetch`
  - `write_back`
  - `spill`
  - `restore`

Retained validation surface:

* `paged_kv_cache_tracks_growth_refill_and_refusal`
* `paged_kv_cache_tracks_owner_bound_page_eviction_and_reclaim`
* `paged_kv_cache_predicts_device_resident_growth_from_empty_seed`
* `paged_kv_cache_predicts_device_resident_growth_from_existing_seed`
* `host_device_kv_residency_reports_prefetch_writeback_and_refusal`

Claim boundary:

* This packet does **not** claim a hidden vLLM-style cache swap.
* It does claim that Psionic already has a real explicit block-manager surface
  with visible geometry, ownership, spill policy, residency accounting, and
  refusal posture.
