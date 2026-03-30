# Psion Executor Local Cluster Ledger

> Status: canonical `PSION-0402` / `#735` record, updated 2026-03-30 after
> landing the first searchable local-cluster ledger for the admitted executor
> lane.

This document records the first cumulative ledger that joins the retained MLX
and 4080 local executor rows into one searchable control-plane surface and now
keeps the first green Mac -> 4080 -> Mac roundtrip closeout bound directly
into the current-best accelerator row.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_local_cluster_ledger_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_local_cluster_ledger_fixtures
```

## What Landed

`psionic-train` now owns one typed local-cluster ledger that wraps the
canonical registration packet, baseline truth packet, and retained roundtrip
closeout packet.

Each retained row now keeps, in one searchable record:

- run id plus search run ids
- admitted profile id
- model id
- candidate status
- frozen eval-pack ids
- config summary
- checkpoint lineage
- cost posture
- metric posture
- retained failure facts
- export status
- recovery status
- roundtrip closure posture

The search index is now explicit for:

- run id
- profile id
- eval-pack id
- model id
- candidate status

That means the admitted MLX and 4080 executor runs are no longer “look in five
packets and remember how they connect.” They now have one cumulative ledger
surface.

The retained rows now also inherit the active mixture version from the
registration packet, so weekly mixture review can stay cumulative instead of
reconstructing mixture identity from separate admission prose.

## Current Retained Truth

- ledger digest:
  `618605effd540810a884fb6797bee683327033cdaae3e79fa5ab0fec51b7b63c`
- registration packet digest:
  `dfad1972f358be079ddd80ac73f5ec85200c16e1e5a708fb11a18bc765cec229`
- baseline truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- roundtrip packet digest:
  `820e605be48dfd4acdef6e1de3e5cd59972c0c7de0894b83f20343a9860f8299`
- MLX ledger row digest:
  `a181d94f23a2ed60b5ece4440beed8f7bea5b69ec9e395baff5d3585d1285410`
- 4080 ledger row digest:
  `7b334f3a3a13b062453e4d6cc06b0f41b330036e6aac6042e7f88a002e4fcc56`
- MLX export status:
  `green`
- MLX recovery status:
  `not_required_same_node`
- 4080 export status:
  `green`
- 4080 recovery status:
  `green`
- 4080 roundtrip closure fact:
  `local_cluster_roundtrip_green`
- shared run-id search key:
  `tailrun-admitted-device-matrix-20260327b`
- shared eval-pack search keys:
  `tassadar.eval.frequent.v0`, `tassadar.eval.promotion.v0`

## Honest Current Posture

The ledger now shows the local-cluster closure split clearly:

- the 4080 row is no longer export-pending
- the green roundtrip closeout is now bound into the current-best row as a
  retained failure-fact successor, not left in separate packet prose
- phase-exit closure is therefore visible directly from the ledger
- promotion still remains a separate question because missing frequent-pack
  eval truth is not solved by a green export loop

## Follow-On Dashboard Surface

The follow-on canonical dashboard packet now lives at:

- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_DASHBOARD.md`
- `fixtures/psion/executor/psion_executor_local_cluster_dashboard_v1.json`

That packet projects the frozen baseline, the retained current-best row, and
the retained candidate row from this ledger instead of inventing a separate
review-only truth source.

The follow-on trace-native metrics packet now lives at:

- `docs/PSION_EXECUTOR_TRACE_NATIVE_METRICS.md`
- `fixtures/psion/executor/psion_executor_trace_native_metrics_v1.json`

That packet keeps the frozen article closeout trio visible per retained
candidate row and per workload, binding trace length, exactness, and
`reference_linear` versus `hull_cache` throughput into this ledger surface.

The follow-on mandatory live-metrics packet now lives at:

- `docs/PSION_EXECUTOR_MANDATORY_LIVE_METRICS.md`
- `fixtures/psion/executor/psion_executor_mandatory_live_metrics_v1.json`

The follow-on failure-bundle taxonomy packet now lives at:

- `docs/PSION_EXECUTOR_FAILURE_BUNDLE_TAXONOMY.md`
- `fixtures/psion/executor/psion_executor_failure_bundle_taxonomy_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_ledger_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_ledger -- --nocapture`
