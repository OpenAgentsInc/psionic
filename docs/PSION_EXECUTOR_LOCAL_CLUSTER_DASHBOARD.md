# Psion Executor Local Cluster Dashboard

> Status: canonical `PSION-0403` / `#736` record, updated 2026-03-30 after
> landing the first baseline-vs-current-best-vs-candidate dashboard packet for
> the admitted executor lane.

This document records the first canonical dashboard packet that projects the
frozen baseline, the retained current-best row, and the retained candidate row
from one shared local-cluster source of truth, and now keeps the retained
roundtrip-green accelerator row visible as the current-best panel.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_local_cluster_dashboard_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_local_cluster_dashboard_fixtures
```

## What Landed

`psionic-train` now owns one typed dashboard packet that sits directly on top
of:

- the frozen baseline-truth record
- the canonical local-cluster run-registration packet
- the searchable local-cluster ledger

The dashboard keeps, in one retained surface:

- one baseline panel for the frozen frequent and promotion pack truth
- one retained current-best run panel
- one retained candidate run panel
- one side-by-side profile comparison strip

Every run-facing panel now keeps:

- metrics
- throughput
- recovery status
- export status
- budget burn

The profile comparison strip keeps both admitted local profiles visible
together under the same shared device-matrix search key, so weekly review can
compare their current posture without reconstructing five packet relationships
by hand.

## Current Retained Truth

- dashboard digest:
  `026da39b01fff5eb4e93025f0a39ad5356c4d8368e603b34b3690e16b140ee28`
- baseline panel digest:
  `7463e54f038488aca1032ef48a8a9cea9d1cf44ab9e5e6d5cfd879613df0e3b1`
- current-best panel digest:
  `b22ee92059c45e6e87bea21c33c444ac9f169c72f1362c2912aedf6749e6a70a`
- candidate panel digest:
  `313ce44d8532b23775384f511f544f02c2656df1d7172d5d7342d13dd54b582d`
- profile-comparison digest:
  `abda7585ebfe5e123b84df4df68d28c88f3c9786c6ffa65174de7c642e55f23d`
- baseline suite counts:
  `11` total, `11` green, `4` manual-review-backed
- retained current-best row:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- retained candidate row:
  `psion_executor_local_cluster_ledger_row_mlx_v1`
- shared run-search key:
  `tailrun-admitted-device-matrix-20260327b`
- candidate-to-current-best step ratio:
  `1.972398532878287`

## Honest Current Posture

The dashboard does not relabel the ledger to make the story cleaner.

Today's retained truth is:

- the 4080 row is the retained `current_best` row
- the MLX row is the retained `candidate` row
- the 4080 row now remains both recovery-green and export-green because the
  Mac -> 4080 -> Mac roundtrip packet is retained directly into the ledger
- the MLX row remains export-green and same-node recovery-not-required

That is the point of this packet. It makes the actual local-cluster state
visible in one place instead of leaving the reviewer to infer it from separate
registration, ledger, export, and decision-grade documents.

## Follow-On Block Surface

The follow-on canonical auto-block report now lives at:

- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_AUTOBLOCKS.md`
- `fixtures/psion/executor/psion_executor_local_cluster_autoblocks_v1.json`

That report turns missing eval, recovery, export, and `reference_linear`
anchor facts into explicit phase-exit and promotion blocks on top of this
dashboard.

The follow-on mandatory live-metrics packet now lives at:

- `docs/PSION_EXECUTOR_MANDATORY_LIVE_METRICS.md`
- `fixtures/psion/executor/psion_executor_mandatory_live_metrics_v1.json`

The follow-on unified throughput packet that keeps dashboard training
throughput and Mac serving throughput in one retained surface now lives at:

- `docs/PSION_EXECUTOR_UNIFIED_THROUGHPUT_REPORTING.md`
- `fixtures/psion/executor/psion_executor_unified_throughput_reporting_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_dashboard_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_dashboard -- --nocapture`
