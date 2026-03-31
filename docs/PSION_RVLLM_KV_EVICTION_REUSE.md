# PSION RVLLM KV Eviction and Reuse

> Status: landed on 2026-03-30 as the eleventh retained RVLLM runtime-harvest
> packet.

This document records the current bounded eviction and reclaimed-page reuse
truth under the admitted paged-KV manager.

The important truth here is not that Psionic now ships a hidden semantic cache
evictor. It does not. The gap was narrower: once paged KV and owner-bound
accounting were explicit, reclaimed-page reuse still was not. Long-context
stress would stay memory-bounded, but logical page identifiers kept growing
monotonically and predictive growth could not report when reclaimed pages were
actually reusable.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_kv_eviction_reuse_v1.json`

Current retained truth:

* packet digest `6054cce13140b995626a55a0b4d182643987d19dd8337e21b39046d3e748f82b`
* explicit eviction strategies:
  - `evict_oldest_pages`
  - `truncate_then_refill`
* explicit reuse strategies:
  - `reclaim_page_index_reuse`
  - `predictive_growth_reuse`
* runtime surfaces:
  - `KvCacheSpillPolicy::EvictOldestPages`
  - `KvCacheOwnershipAccounting.reclaimed_pages`
  - `KvCacheOwnershipAccounting.reused_pages`
  - `InMemoryKvCache::reusable_page_indices`

Retained long-context stress rows:

| Scenario | Peak live pages before | Peak live pages after | Max page index before | Max page index after | Reuse hits before | Reuse hits after | Tail correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `session_ring_18_tokens` | `3` | `3` | `8` | `2` | `0` | `6` | `true` |
| `truncate_refill_window` | `2` | `2` | `2` | `1` | `0` | `1` | `true` |

Retained validation surface:

* `paged_kv_cache_tracks_owner_bound_page_eviction_and_reclaim`
* `paged_kv_cache_reuses_reclaimed_page_indices_under_long_context_stress`
* `paged_kv_cache_predicts_reused_page_growth_from_existing_reclaim`
* `host_device_kv_residency_reports_prefetch_writeback_and_refusal`

Claim boundary:

* This packet claims only bounded oldest-page eviction plus deterministic
  reclaimed-page reuse under the existing paged-KV manager.
* It does **not** claim hidden semantic eviction, unbounded long-context
  scaling, or a broad swap daemon beyond the already-explicit spill and
  residency surfaces.
