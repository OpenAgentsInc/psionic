# Psion Actual Pretraining Evidence Contract

> Status: canonical output, evidence, provenance, and redaction contract for
> the actual `Psion` pretraining lane, written 2026-04-02 after freezing the
> shared retained layout above the recipe and topology bundles.

This document freezes one operator-owned output and evidence family for the
canonical actual pretraining lane.

It does not implement backup, auto-resume, automatic evaluation, or dashboards
by itself. It does fix the layout and naming that those later issues must
reuse.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_evidence_contract.rs`
  owns the machine-readable contract.
- `crates/psionic-train/examples/psion_actual_pretraining_evidence_contract_fixtures.rs`
  regenerates the committed fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_evidence_contract_v1.json`
  is the canonical evidence contract fixture.

Stable schema version:

- `psion.actual_pretraining_evidence_contract.v1`

## Frozen Family

The contract freezes:

- contract id: `psion_actual_pretraining_evidence_contract_v1`
- lane id: `psion_actual_pretraining_v1`
- evidence family: `psion.actual_pretraining.evidence.v1`
- run-root family: `psion_actual_pretraining_runs/<run_id>`

It also binds directly to the already-frozen recipe and topology/storage
bundles rather than redefining them.

## Example Output Tree

Every actual-lane run must fit this family:

```text
psion_actual_pretraining_runs/<run_id>/
  manifests/
    launch_manifest.json
    resume_manifest.json
  preflight/
    hardware_qualification.json
  status/
    current_run_status.json
    retained_summary.json
  checkpoints/
    latest_accepted_checkpoint_pointer.json
    step-<optimizer_step>/
      checkpoint_manifest.json
  continuation/
    accepted_checkpoint_handoff.json
  evals/
    checkpoint_eval_step-<optimizer_step>.json
  exports/
    promoted_checkpoint_export_manifest.json
  logs/
    launcher.log
  alerts/
    latest_redacted_alert.json
  closeout/
    closeout_bundle.json
```

The point is not that every later issue writes every file immediately. The
point is that later launcher, backup, eval, export, alerting, and closeout
work all write into one shared family.

The typed status and retained-summary surfaces that occupy the `status/`
directory are now frozen separately in
`docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`.

## Required Provenance

The contract requires at least these fields:

- `git_commit_sha`
- `selected_git_ref`
- `dirty_tree_admission`

Those fields must appear in the retained launch manifest and repeat in the
closeout bundle so final operator claims stay tied to one exact source state.

An optional `workspace_status_sha256` field is also reserved for launchers that
materialize from a local checkout and want to retain a digest of the status
snapshot without copying the raw status payload everywhere.

## Redaction Rules

Retained artifacts must keep:

- refs
- digests
- env-var names
- cluster labels
- topology digests
- redacted host labels

Retained artifacts must not keep:

- raw credential payloads
- access tokens
- private keys
- service-account JSON blobs
- private IPs
- raw SSH targets
- raw secret bucket material

That applies to manifests, preflight receipts, logs, and alerts. The retained
surfaces should point to credentials by declared source name and digest, not by
copying secret content or raw connection details.

## Why This Matters

Without one frozen evidence family, later launcher, resume, eval, export, and
closeout work would each invent slightly different filenames, provenance
fields, and redaction rules.

This contract prevents that drift before the actual-lane launcher starts
writing those retained artifacts for real.

The first concrete writer for this family now exists in
`./TRAIN --lane actual_pretraining start|resume`. It currently writes the
launch or resume manifest, retained hardware qualification receipt, retained
status surfaces, canonical checkpoint pointer, launcher log, and a provisional
closeout bundle that repeats git provenance early. Resume over an accepted
checkpoint also writes the retained continuation handoff at
`continuation/accepted_checkpoint_handoff.json`, which binds that accepted
checkpoint to the frozen `general_sft -> agentic_sft` continuation target.
Later hardening issues extend the same retained family instead of replacing
it.
