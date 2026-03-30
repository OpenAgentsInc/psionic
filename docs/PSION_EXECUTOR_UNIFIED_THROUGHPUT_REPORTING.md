# Psion Executor Unified Throughput Reporting

> Status: canonical `PSION-0605` / `#751` record, updated 2026-03-30 after
> landing the first retained throughput surface that keeps training and
> fast-route serving truth together for the admitted executor lane.

This document records the first canonical throughput packet that binds the
local-cluster training dashboard to the retained Mac export and fast-route
truth. It makes replacement-throughput review one retained surface instead of
splitting it between training-only rows and separate serving-only receipts.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_unified_throughput_reporting_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_unified_throughput_reporting_fixtures
```

## What Landed

`psionic-train` now owns one typed unified-throughput packet that binds:

- the canonical local-cluster dashboard packet
- the retained Mac export-inspection packet
- the retained current-best 4080 training-throughput row
- the retained candidate MLX training-throughput row
- the retained admitted `hull_cache` serving-throughput floor
- one explicit replacement blocker row for serving-throughput regression

That means the admitted executor lane now has one reviewable answer to:

- what the current-best training throughput is
- what the candidate training throughput is
- what the admitted serving-throughput floor is
- whether the replacement candidate is blocked on serving-throughput truth

## Current Retained Truth

- report digest:
  `ff12ece15c7917e2c430cb139d81c36c2d9e2964f9ee8197275664314fc037a7`
- dashboard digest:
  `026da39b01fff5eb4e93025f0a39ad5356c4d8368e603b34b3690e16b140ee28`
- export-inspection digest:
  `9d6a39d78400f4a0c6c86398b677b9880080e8351653b3f68ccadb6e4a06aa8a`
- current-best training row:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- current-best training profile:
  `local_4080_cuda_tailnet_x86_64`
- current-best training throughput:
  `82.402520498292` steps/sec,
  `2357371.30641513` source tokens/sec
- candidate training row:
  `psion_executor_local_cluster_ledger_row_mlx_v1`
- candidate training profile:
  `local_mac_mlx_aarch64`
- candidate training throughput:
  `162.530610536304` steps/sec,
  `4649675.706222572` source tokens/sec
- candidate-to-current-best step ratio:
  `1.972398532878287`
- serving machine class:
  `host_cpu_aarch64`
- serving anchor metric:
  `tassadar.reference_linear_steps_per_second`
- serving fast-route metric:
  `tassadar.hull_cache_steps_per_second`
- serving throughput-floor digest:
  `b500d330f5146399b4b49f054e8ebd45aa584707f8f66b0a3433424ccbfa086d`
- hull-cache closure digest:
  `582b36210e020462e1d52844d3de28aff0c7beed5b1b73eb856af1a110631c9b`
- minimum retained `hull_cache` speedup over `reference_linear`:
  `1.690977509006`
- maximum retained `hull_cache` remaining gap versus CPU reference:
  `2.683604159673`
- replacement candidate row:
  `psion_executor_local_cluster_ledger_row_mlx_v1`
- replacement blocked:
  `false`
- active replacement block ids:
  none
- serving-throughput block row:
  `serving_throughput_regression_candidate -> green_serving_throughput_floor`

## Honest Current Meaning

The retained packet does not collapse training and serving into one number.

It does something more useful:

- it keeps the 4080 current-best training row and the MLX candidate training
  row visible together
- it keeps the admitted serving floor explicit on the Mac replacement surface
- it makes replacement blockage machine-readable if serving throughput ever
  regresses even while training throughput looks good

Right now that gate is green. The candidate leads on training throughput and
the admitted serving floor remains green, so serving throughput does not block
replacement.

That is the actual closeout: replacement is no longer allowed to ignore
serving-throughput truth.

The follow-on retained long-run rehearsal packet that now uses this green
replacement gate inside one full interruption-to-review rehearsal lives at:

- `docs/PSION_EXECUTOR_LONG_RUN_REHEARSAL.md`
- `fixtures/psion/executor/psion_executor_long_run_rehearsal_v1.json`

## Follow-On Surfaces

The dashboard packet that now feeds the training side of this report lives at:

- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_DASHBOARD.md`
- `fixtures/psion/executor/psion_executor_local_cluster_dashboard_v1.json`

The export-inspection packet that now feeds the serving side of this report
lives at:

- `docs/PSION_EXECUTOR_MAC_EXPORT_INSPECTION.md`
- `fixtures/psion/executor/psion_executor_mac_export_inspection_v1.json`

The phase-exit and promotion auto-block surface that this report now extends
for replacement review lives at:

- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_AUTOBLOCKS.md`
- `fixtures/psion/executor/psion_executor_local_cluster_autoblocks_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_unified_throughput_reporting_fixtures`
- `cargo test -q -p psionic-train psion_executor_unified_throughput_reporting -- --nocapture`
