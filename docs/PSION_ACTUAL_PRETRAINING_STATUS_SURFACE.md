# Psion Actual Pretraining Status Surface

> Status: canonical current-status and retained-summary surfaces for the actual
> `Psion` pretraining lane, written 2026-04-02 after freezing the status
> contract ahead of the real launcher implementation.

This document freezes one current-status artifact, one retained-summary
artifact, and one status command for the actual pretraining lane.

It does not claim that the full actual-lane launcher already exists. It does
fix the named launcher surfaces that later work must implement and the retained
status files that later launcher work must write.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_status_surface.rs` owns
  the typed status and retained-summary contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_status_surface_fixtures.rs`
  regenerates the committed fixtures and example run-root tree.
- `scripts/psion-actual-pretraining-status.sh` is the canonical operator
  status command for reading those retained files from a run root.
- `fixtures/psion/pretrain/psion_actual_pretraining_current_run_status_v1.json`
  is the canonical current-status fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_retained_summary_v1.json`
  is the canonical retained-summary fixture.

## Reserved Launcher Surfaces

The actual-lane launcher contract now reserves four named surfaces:

- `psion_actual_pretraining.start`
- `psion_actual_pretraining.dry_run`
- `psion_actual_pretraining.resume`
- `psion_actual_pretraining.status`

`#828` will implement the actual `start` and `resume` paths. This issue fixes
their names now so later launcher work does not invent them ad hoc.

## Retained Status Paths

Inside an actual-lane run root, the retained status files are:

- `status/current_run_status.json`
- `status/retained_summary.json`

Those paths are already reserved by the evidence contract in
`docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`.

## Status Command

The canonical status command is:

```bash
./scripts/psion-actual-pretraining-status.sh --run-root <path>
```

It reads the two retained status files and prints the last known run state:

- run id
- phase
- last completed step
- latest checkpoint label
- selected git ref
- git commit SHA
- dirty-tree admission posture
- status surface id

## Why This Matters

The bounded reference-pilot launcher already had operator-manifest and summary
surfaces. The actual lane needed the same kind of named retained status surface
before the real launcher implementation starts.

Freezing the status and retained-summary artifacts now means `#828` can
implement the actual launcher against one explicit contract instead of choosing
new status names later.
