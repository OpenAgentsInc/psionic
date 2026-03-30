# Psion Executor MLX Smoke Run

> Status: canonical `PSION-0203` / `#722` retained MLX smoke-run packet for
> the executor lane, updated 2026-03-30.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_mlx_smoke_run_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mlx_smoke_run_fixtures
```

## What Landed

The executor lane now has one typed MLX smoke-run packet built from the
retained same-node MLX training/export lane plus one explicit frequent-pack
subset.

Retained sources:

- smoke report:
  `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/report.json`
- durable bundle:
  `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/portable_bundle.safetensors`
- checkpoint prerequisite:
  `fixtures/psion/executor/psion_executor_mlx_checkpoint_compatibility_v1.json`
- frozen frequent pack:
  `fixtures/psion/executor/psion_executor_eval_packs_v1.json`

## Approved Frequent-Pack Subset

Phase-one MLX smoke still uses an admitted subset instead of pretending the Mac
already closes the full executor-model training objective.

The committed subset is:

- subset id: `tassadar.eval.frequent.v0::mlx_smoke_subset_v1`
- parent pack: `tassadar.eval.frequent.v0`
- suite id: `frequent_operator_review_cases_v0`

Included cases that must stay green:

- `artifact_packet_complete`
- `checkpoint_restore_rehearsal_green`
- `export_smoke_green`

Deferred case:

- `local_cluster_roundtrip_green`

That deferred case is intentionally held for EPIC 3 because it belongs to the
Mac -> 4080 -> Mac control-plane lane rather than the first Mac-only MLX smoke
run.

## Honest Boundary

This packet proves that the Mac is now running a real bounded MLX smoke lane
with:

- a durable checkpoint/export artifact
- explicit ledger-style packet facts
- an admitted `tassadar.eval.frequent.v0` subset

It does not claim full executor-model MLX training closure yet. The current
smoke objective is an executor-lane admission surrogate built on the retained
MLX same-node substrate.
