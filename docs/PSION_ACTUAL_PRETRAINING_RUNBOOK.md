# Psion Actual Pretraining Runbook

Status: canonical operator runbook for the explicit actual `Psion`
pretraining lane, written 2026-04-02 when the repo landed the first real
start, dry-run, resume, and status command for `psion_actual_pretraining_v1`.

## What This Runbook Is For

This runbook exists so the repo has one explicit operator path for the actual
broader-pretraining lane without changing the default meaning of `./TRAIN`.

The commands are:

```bash
./TRAIN --lane actual_pretraining start [options]
./TRAIN --lane actual_pretraining record-checkpoint --run-root <path> --checkpoint-label <label> --optimizer-step <step> --checkpoint-ref <ref> [options]
./TRAIN --lane actual_pretraining backup --run-root <path> [options]
./TRAIN --lane actual_pretraining resume --run-root <path> [options]
./TRAIN --lane actual_pretraining status --run-root <path>
./TRAIN --lane actual_pretraining dashboard --run-root <path>
```

`./TRAIN` without `--lane actual_pretraining` still means the bounded
reference-pilot lane.

## Current Claim Boundary

The actual-lane command now does these things for real:

- loads the frozen actual-lane spec, recipe bundle, scaling bundle, data
  bundle, baseline-tools bundle, systems bundle, topology/storage bundle,
  evidence contract, and status surface contract directly from committed repo
  artifacts
- writes `preflight/hardware_qualification.json` before launch or resume
- writes `preflight/run_shape_qualification.json` before launch or resume
- refuses non-dry-run start or resume when either retained preflight receipt is
  not admitted
- refuses dirty launches by default unless `--allow-dirty-tree` is supplied
- resolves and retains the selected git ref plus exact commit SHA
- writes the canonical launch or resume manifest under the retained evidence
  family
- writes the canonical current-status and retained-summary files
- writes the canonical latest-checkpoint pointer file
- writes accepted checkpoint manifests plus durable-backup receipts
- writes auto-resume receipts and retained stale/corrupt-pointer recovery drills
- writes automatic checkpoint-eval decisions on accepted checkpoints
- can retain checkpoint-eval retry receipts plus a redacted alert when the eval
  worker is unavailable
- writes one retained dashboard packet plus one retained aggregate active-alert
  feed
- can inject a failed-upload refusal drill without manual artifact editing
- repeats provenance into the provisional closeout bundle
- exposes the canonical status command
- exposes the canonical dashboard command

It does not yet claim:

- external alert delivery or paging
- a cluster-connected streaming dashboard
- completed distributed cluster execution

Those come later in the roadmap. This launcher is the operator contract, not
the full hardening pass.

## Start Command

Dry-run materialization:

```bash
./TRAIN --lane actual_pretraining start --dry-run
```

Dry-run with an admitted retained observation snapshot:

```bash
./TRAIN --lane actual_pretraining start \
  --dry-run \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json
```

Start with an explicit run id and output root:

```bash
./TRAIN --lane actual_pretraining start \
  --run-id run-psion-actual-20260402t120000z \
  --output-root ~/scratch/psion_actual_pretraining_runs/run-psion-actual-20260402t120000z
```

Default output root:

- `~/scratch/psion_actual_pretraining_runs/<run_id>`

The start path writes:

- `manifests/launch_manifest.json`
- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `preflight/hardware_qualification.json`
- `preflight/run_shape_qualification.json`
- `closeout/closeout_bundle.json`
- `dashboard/current_dashboard.json`
- `alerts/active_alerts.json`
- `logs/launcher.log`

Non-dry-run `start` now refuses when either preflight receipt lands with
`admission_state = refused`.

Before the first accepted checkpoint exists, the retained state is explicit:

- phase: `dry_run_planned` or `launch_staged`
- latest checkpoint label: `pending_first_checkpoint`
- last completed step: `0`

## Record Checkpoint

Canonical accepted-checkpoint materialization:

```bash
./TRAIN --lane actual_pretraining record-checkpoint \
  --run-root <path> \
  --checkpoint-label broader-pretrain-final \
  --optimizer-step 16384 \
  --checkpoint-ref checkpoint://psion/broad/pretrain/final
```

This command promotes the run from `pending_first_checkpoint` into one accepted
checkpoint lineage. It writes:

- `checkpoints/step-<optimizer_step>/checkpoint_manifest.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `checkpoints/latest_accepted_checkpoint_backup_receipt.json`
- `checkpoints/backups/latest_accepted_checkpoint_pointer.backup.json`
- `checkpoints/backups/step-<optimizer_step>/checkpoint_manifest.backup.json`
- `evals/checkpoint_eval_step-<optimizer_step>.json`
- `evals/latest_checkpoint_eval_decision.json`
- refreshed status, retained-summary, closeout, and launcher-log surfaces
- refreshed retained dashboard and active-alert feed

The default checkpoint byte count comes from the frozen systems bundle. The
default checkpoint object digest is a stable synthetic digest over the accepted
checkpoint identity unless the operator provides an explicit digest.

Unavailable-worker rehearsal:

```bash
./TRAIN --lane actual_pretraining record-checkpoint \
  --run-root <path> \
  --checkpoint-label broader-pretrain-final \
  --optimizer-step 16384 \
  --checkpoint-ref checkpoint://psion/broad/pretrain/final \
  --inject-eval-worker-unavailable
```

That path writes:

- `evals/checkpoint_eval_failure_step-<optimizer_step>.json`
- `evals/latest_checkpoint_eval_failure.json`
- `alerts/latest_redacted_alert.json`

It keeps the checkpoint itself admitted and backed up, but retains explicit
retry-required evidence instead of silently skipping automatic eval.

## Backup Command

Canonical durable-backup replay:

```bash
./TRAIN --lane actual_pretraining backup --run-root <path>
```

This command rereads the current accepted pointer and checkpoint manifest and
re-materializes the retained backup family plus
`checkpoints/latest_accepted_checkpoint_backup_receipt.json`.
It also refreshes the retained status, closeout, dashboard, and active-alert
surfaces so backup refusal or success becomes operator-visible without reading
raw receipts directly first.

Failure-injection rehearsal:

```bash
./TRAIN --lane actual_pretraining backup \
  --run-root <path> \
  --inject-failed-upload
```

That drill writes:

- a refused backup receipt
- `checkpoints/failures/failed_upload_drill.json`

It retains declared secret/config source names only. Raw SSH, bucket, or
service-account payloads are not copied into retained artifacts or logs.

## Resume Command

Canonical resume:

```bash
./TRAIN --lane actual_pretraining resume --run-root <path>
```

Resume first reads:

- `<run-root>/checkpoints/latest_accepted_checkpoint_pointer.json`

If that primary pointer is stale or corrupt, resume falls back to:

- `<run-root>/checkpoints/latest_accepted_checkpoint_backup_receipt.json`
- `<run-root>/checkpoints/backups/latest_accepted_checkpoint_pointer.backup.json`
- `<run-root>/checkpoints/backups/step-<optimizer_step>/checkpoint_manifest.backup.json`

Resume refuses when neither the primary pointer nor the retained backup family
can produce an admitted accepted checkpoint. It also writes and consumes
`preflight/hardware_qualification.json` plus
`preflight/run_shape_qualification.json`, and non-dry-run resume refuses when
either receipt is not admitted. Every resume attempt now writes:

- `checkpoints/auto_resume_receipt.json`

Corrupt or stale primary-pointer recovery also writes:

- `checkpoints/failures/corrupt_pointer_drill.json`
- or `checkpoints/failures/stale_pointer_drill.json`

When resume succeeds, it writes:

- `manifests/resume_manifest.json`
- refreshed status and retained-summary files
- `continuation/accepted_checkpoint_handoff.json`
- refreshed provisional closeout bundle
- appended `logs/launcher.log`

The continuation handoff binds the accepted checkpoint to the frozen
`general_sft -> agentic_sft` target and preserves the bounded plugin
benchmark-pack bindings already attached to that target. It does not claim that
the continuation stage has already run.

If auto-resume cannot select a valid checkpoint, the command still retains an
explicit `auto_resume_receipt.json` with `resolution_state = refused` and logs
`phase=resume_refused_auto_resume` instead of leaving the run root ambiguous.

## Dashboard Command

```bash
./TRAIN --lane actual_pretraining dashboard --run-root <path>
```

This is a thin wrapper over
`scripts/psion-actual-pretraining-dashboard.sh`. It reads:

- `dashboard/current_dashboard.json`
- `alerts/active_alerts.json`

and prints:

- run id
- phase
- git provenance
- throughput posture
- loss and gradient visibility posture
- checkpoint backup and eval posture
- hardware health posture
- active alert count plus the highest severity
- one line per active alert when any alert is present

## Status Command

```bash
./TRAIN --lane actual_pretraining status --run-root <path>
```

This is a thin wrapper over `scripts/psion-actual-pretraining-status.sh`. It
prints:

- run id
- phase
- last completed step
- latest checkpoint label
- selected git ref
- git commit SHA
- dirty-tree admission posture
- status surface id
- latest checkpoint-eval decision and score when present
- latest checkpoint-eval failure and latest alert when present

## Dirty Trees And Provenance

Dirty working trees are refused by default.

If an operator deliberately overrides that rule:

```bash
./TRAIN --lane actual_pretraining start --allow-dirty-tree --dry-run
```

the launcher records:

- `dirty_tree_admission = allowed_by_operator_override`
- `workspace_status_sha256`

It still retains:

- `selected_git_ref`
- `git_commit_sha`

Those fields appear in the launch or resume manifest and repeat in the
provisional closeout bundle.

## Related Docs

- `docs/PSION_ACTUAL_PRETRAINING_LANE.md`
- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SCALING_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_BASELINE_TOOLS_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_DATA_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_ACTUAL_PRETRAINING_CHECKPOINT_EVALS.md`
- `docs/PSION_ACTUAL_PRETRAINING_HARDWARE_QUALIFICATION.md`
- `docs/PSION_ACTUAL_PRETRAINING_RUN_SHAPE_QUALIFICATION.md`
- `docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SYSTEMS_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF.md`
- `docs/PSION_ACTUAL_PRETRAINING_CHECKPOINT_RECOVERY.md`
