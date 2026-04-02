# Psion Actual Pretraining Checkpoint Recovery

> Status: canonical checkpoint-backup and auto-resume contract for the actual
> `Psion` pretraining lane, written 2026-04-02 after landing the retained
> manifest, backup-receipt, auto-resume, and failure-drill surfaces.

This document records the actual-lane checkpoint lifecycle that now exists in
`psionic`.

It is narrower than the older generic checkpoint references. This doc is only
about the canonical `psion_actual_pretraining_v1` operator lane and the
artifacts that `./TRAIN --lane actual_pretraining` now writes.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_checkpoint_recovery.rs`
  owns the typed checkpoint manifest, backup receipt, auto-resume receipt, and
  failure-drill contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_operator.rs` owns
  the real operator flow that writes and consumes those contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_checkpoint_recovery_fixtures.rs`
  regenerates the committed recovery fixtures and rehearsal example run roots.
- `scripts/train-psion-actual-pretraining.sh` is the operator entrypoint for
  the checkpoint lifecycle commands.

Committed fixtures:

- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_manifest_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_backup_receipt_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_auto_resume_receipt_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_failure_drill_failed_upload_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_failure_drill_corrupt_pointer_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_failure_drill_stale_pointer_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_recovery_example/`

## Operator Commands

The actual-lane checkpoint lifecycle now has four operator actions:

```bash
./TRAIN --lane actual_pretraining start [options]
./TRAIN --lane actual_pretraining record-checkpoint --run-root <path> --checkpoint-label <label> --optimizer-step <step> --checkpoint-ref <ref> [options]
./TRAIN --lane actual_pretraining backup --run-root <path> [options]
./TRAIN --lane actual_pretraining resume --run-root <path> [options]
```

`record-checkpoint` is the first surface that promotes one retained
`pending_first_checkpoint` run into an accepted checkpoint lineage. It writes:

- `checkpoints/step-<optimizer_step>/checkpoint_manifest.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `checkpoints/latest_accepted_checkpoint_backup_receipt.json`
- `checkpoints/backups/latest_accepted_checkpoint_pointer.backup.json`
- `checkpoints/backups/step-<optimizer_step>/checkpoint_manifest.backup.json`

It also updates:

- `status/current_run_status.json`
- `status/retained_summary.json`
- `closeout/closeout_bundle.json`
- `logs/launcher.log`

`backup` replays the durable-backup step for the current accepted checkpoint.
It rewrites the backup receipt and backup copies and, when asked to inject a
failure drill, also writes one retained refusal receipt under
`checkpoints/failures/`.

`resume` now performs zero-guess checkpoint selection. It first tries the
primary pointer at `checkpoints/latest_accepted_checkpoint_pointer.json`. If
that pointer is stale or corrupt, it falls back to the retained backup family
and restores the primary pointer from:

- `checkpoints/latest_accepted_checkpoint_backup_receipt.json`
- `checkpoints/backups/latest_accepted_checkpoint_pointer.backup.json`
- `checkpoints/backups/step-<optimizer_step>/checkpoint_manifest.backup.json`

Every resume attempt now writes:

- `checkpoints/auto_resume_receipt.json`

Corrupt or stale primary-pointer recovery also writes:

- `checkpoints/failures/corrupt_pointer_drill.json`
- or `checkpoints/failures/stale_pointer_drill.json`

## Retained Contracts

The recovery surface freezes four schema families:

- `psion.actual_pretraining_checkpoint_manifest.v1`
- `psion.actual_pretraining_checkpoint_backup_receipt.v1`
- `psion.actual_pretraining_auto_resume_receipt.v1`
- `psion.actual_pretraining_checkpoint_failure_drill.v1`

Those artifacts always retain:

- lane id
- run id
- selected git ref
- exact git commit SHA
- dirty-tree admission posture
- optional `workspace_status_sha256` when dirty-tree override is used
- redacted credential-source names instead of raw credential payloads

The backup and auto-resume surfaces now bind checkpoint recovery to the same
evidence family described in
`docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md` instead of inventing a
second storage or naming scheme.

## Failure Drills

The retained failure-drill family now covers three cases:

- `failed_upload`
  `backup --inject-failed-upload` writes a refused backup receipt plus a
  failure-drill artifact without copying any secret payload into retained
  evidence.
- `corrupt_pointer`
  `resume` can recover from invalid primary-pointer JSON by restoring the
  primary pointer from the retained backup family and recording that recovery.
- `stale_pointer`
  `resume` can recover when the primary pointer no longer references a valid
  retained checkpoint manifest and records that repair without manual editing.

If neither the primary pointer nor an admitted backup receipt can produce a
valid checkpoint selection, `resume` still writes an
`auto_resume_receipt.json`, but it records `resolution_state = refused` rather
than pretending the launcher made progress.

## Secrets And Provenance

The backup receipt keeps only declared credential-source names such as the
actual-lane bucket URL or secret-file env binding. The operator path does not
copy raw SSH, object-store, or service-account payloads into manifests, backup
receipts, auto-resume receipts, failure drills, or launcher logs.

The launcher-side git interactions remain read-only:

- `git rev-parse`
- `git status --porcelain=v1 --untracked-files=normal`

The actual-lane checkpoint lifecycle does not use `git checkout`, `git reset`,
or any destructive workspace mutation to stage a backup or a resume.

## Claim Boundary

This surface now proves:

- one accepted checkpoint can be recorded under the actual-lane evidence family
- one durable backup receipt can bind that checkpoint to the retained backup
  family
- resume can recover from stale or corrupt primary pointers without manual
  editing
- failed uploads retain explicit refusal evidence instead of silent optimism

It still does not prove:

- automatic checkpoint eval
- dashboard or alert routing
- final continue-vs-restart policy
- completed cluster-scale broader-pretraining execution

## Related Docs

- `docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SYSTEMS_BUNDLE.md`
- `docs/TRAIN_CHECKPOINT_RECOVERY_REFERENCE.md`
