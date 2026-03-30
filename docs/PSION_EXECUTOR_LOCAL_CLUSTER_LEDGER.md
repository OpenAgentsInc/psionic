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

## Current Retained Truth

- ledger digest:
  `9b86949597220f5bb4eb80c2b313fae2416c1908771ea3ae9771ec3084d06dd3`
- registration packet digest:
  `cc500567c6570ae383bf770d9a8d6c732025cc29f4c6ff99741b8cd0aa1e7474`
- baseline truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- roundtrip packet digest:
  `dc2d4cc82a4b5b952032991ca5786c8c2aadd4bb6155801d85c73c07798a1ef1`
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

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_ledger_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_ledger -- --nocapture`
