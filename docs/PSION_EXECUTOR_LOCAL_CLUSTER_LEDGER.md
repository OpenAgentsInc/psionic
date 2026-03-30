# Psion Executor Local Cluster Ledger

> Status: canonical `PSION-0402` / `#735` record, updated 2026-03-30 after
> landing the first searchable local-cluster ledger for the admitted executor
> lane.

This document records the first cumulative ledger that joins the retained MLX
and 4080 local executor rows into one searchable control-plane surface.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_local_cluster_ledger_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_local_cluster_ledger_fixtures
```

## What Landed

`psionic-train` now owns one typed local-cluster ledger that wraps the
canonical registration packet and baseline truth packet.

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
  `1650d362d9ea49099aaad6dc94459eb5530e5f35b19e74492ec47f9b5be0f632`
- registration packet digest:
  `cc500567c6570ae383bf770d9a8d6c732025cc29f4c6ff99741b8cd0aa1e7474`
- baseline truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- MLX ledger row digest:
  `a181d94f23a2ed60b5ece4440beed8f7bea5b69ec9e395baff5d3585d1285410`
- 4080 ledger row digest:
  `78d08622a1daee7e4b4913b59e82a3081354edd6a2ad0dc3242ced2ddd07630e`
- MLX export status:
  `green`
- MLX recovery status:
  `not_required_same_node`
- 4080 export status:
  `pending_mac_roundtrip_validation`
- 4080 recovery status:
  `green`
- shared run-id search key:
  `tailrun-admitted-device-matrix-20260327b`
- shared eval-pack search keys:
  `tassadar.eval.frequent.v0`, `tassadar.eval.promotion.v0`

## Honest Current Gap

The ledger is now searchable and cumulative, but it still shows one live gap
explicitly:

- the 4080 row has checkpoint and recovery truth
- the 4080 row does **not** yet have Mac-side export validation
- that is why export remains `pending_mac_roundtrip_validation`

This is deliberate. The ledger is supposed to show the current state, not hide
the next missing proof.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_ledger_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_ledger -- --nocapture`
