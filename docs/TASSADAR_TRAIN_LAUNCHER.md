# Tassadar Train Launcher

`./TRAIN_TASSADAR` is the operator-facing launcher for the retained Tassadar
training lanes that already have explicit checker paths.

It now supports:

```bash
./TRAIN_TASSADAR start
./TRAIN_TASSADAR dry-run --lane tassadar_sudoku_v0_promotion_v3
./TRAIN_TASSADAR status --run-root tassadar_operator_runs/run-tassadar-20260402t200000z
```

The launcher writes retained outputs under one operator run root using the same
top-level pattern as the Psion actual lane:

- `manifests/launch_manifest.json`
- `status/current_run_status.json`
- `status/retained_summary.json`

The committed canonical launcher fixtures live at:

- `fixtures/tassadar/operator/tassadar_train_launch_manifest_v1.json`
- `fixtures/tassadar/operator/tassadar_train_current_run_status_v1.json`
- `fixtures/tassadar/operator/tassadar_train_retained_summary_v1.json`

Example retained launcher roots live at:

- `fixtures/tassadar/operator/tassadar_train_launcher_example/start/run-tassadar-20260402t200000z`
- `fixtures/tassadar/operator/tassadar_train_launcher_example/dry-run/run-tassadar-20260402t200000z`

## Supported Lanes

- `tassadar_article_transformer_trace_bound_trained_v0`
  This is the canonical default lane from `docs/TASSADAR_DEFAULT_TRAIN_LANE.md`.
  Checker:
  `scripts/check-tassadar-default-train-lane.sh`
- `tassadar_hungarian_10x10_article_learned_v0`
  This is the exact learned article benchmark lane.
  Checker:
  `scripts/check-tassadar-acceptance.sh`
- `tassadar_sudoku_v0_promotion_v3`
  This is the retained learned 4x4 promotion bundle.
  Checker:
  `scripts/check-tassadar-4x4-promotion-gate.sh fixtures/tassadar/runs/sudoku_v0_promotion_v3`

## Deliberate Exclusions

The launcher does not pretend every historical lane is operator-ready yet.

- The bounded 9x9 learned reference lane is still omitted because it does not
  yet have a separate operator-owned checker path on the same footing.
- The later 4080 executor decision-grade and replacement tracks remain later
  candidate programs above the incumbent default lane.

## Claim Boundary

The launcher makes lane selection, checker choice, and retained operator output
roots explicit. It does not claim that all Tassadar internals are unified or
that selecting a lane automatically executes or promotes it.
