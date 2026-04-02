# Psion Actual Pretraining Status Surface

> Status: canonical current-status and retained-summary surfaces for the actual
> `Psion` pretraining lane, written 2026-04-02 after freezing the status
> contract ahead of the real launcher implementation.

This document freezes one current-status artifact, one retained-summary
artifact, and one status command for the actual pretraining lane.

The actual-lane launcher now writes those retained files through
`./TRAIN --lane actual_pretraining start|resume|record-checkpoint|decide-continue-restart|rehearse-base-lane`,
and the same status command reads them back from a run root.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_status_surface.rs` owns
  the typed status and retained-summary contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_status_surface_fixtures.rs`
  regenerates the committed fixtures and example run-root tree.
- `scripts/psion-actual-pretraining-status.sh` is the canonical operator
  status command for reading those retained files from a run root.
- `scripts/psion-actual-pretraining-dashboard.sh` is the canonical operator
  dashboard reader for the retained observability packet written beside the
  status files.
- `fixtures/psion/pretrain/psion_actual_pretraining_current_run_status_v1.json`
  is the canonical current-status fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_retained_summary_v1.json`
  is the canonical retained-summary fixture.

## Reserved Launcher Surfaces

The actual-lane launcher contract now uses four named surfaces:

- `psion_actual_pretraining.start`
- `psion_actual_pretraining.dry_run`
- `psion_actual_pretraining.resume`
- `psion_actual_pretraining.status`

Those names are now wired into the actual-lane start, dry-run, resume, and
status paths so later hardening work can build on one stable contract.

## Retained Status Paths

Inside an actual-lane run root, the retained status files are:

- `status/current_run_status.json`
- `status/retained_summary.json`

Those paths are already reserved by the evidence contract in
`docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`.

The status artifacts now also retain the reserved continuation handoff path:

- `continuation/accepted_checkpoint_handoff.json`

That keeps the last-known operator state and the accepted-checkpoint handoff in
the same named family when resume has selected an accepted checkpoint.

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

Pre-first-checkpoint launch states are now explicit:

- `dry_run_planned`
- `launch_staged`

Those phases legitimately retain `last_completed_step = 0` and
`latest_checkpoint_label = pending_first_checkpoint`.

Checkpoint-lifecycle and refusal states now also appear in the retained status
family when the operator path advances beyond first launch:

- `resume_dry_run_planned`
- `resume_staged`
- `checkpoint_evaluated`
- `checkpoint_eval_retry_required`
- `checkpoint_backed_up`
- `checkpoint_backup_refused`
- `continue_decision_recorded`
- `hold_decision_recorded`
- `restart_decision_recorded`
- `base_lane_rehearsal_complete`
- `resume_refused_auto_resume`

## Why This Matters

The bounded reference-pilot launcher already had operator-manifest and summary
surfaces. The actual lane needed the same kind of named retained status surface
so the actual launcher could materialize one honest operator bundle without
inventing ad hoc filenames. The same status command now also prints the latest
checkpoint-eval decision and score when present, the latest retry-required
failure and redacted alert when automatic eval could not run, and the latest
continue-restart decision plus operator action when that retained surface has
been written. When the base-lane proof gate runs, the same status surface ends
at `phase = base_lane_rehearsal_complete` instead of leaving closeout state
hidden inside the closeout bundle only.

The remaining hardening work now extends these retained surfaces instead of
defining them. The companion retained dashboard and active-alert feed now live
in `docs/PSION_ACTUAL_PRETRAINING_DASHBOARD_AND_ALERTS.md`.
