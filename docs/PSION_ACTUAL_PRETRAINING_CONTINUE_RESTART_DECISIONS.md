# Psion Actual Pretraining Continue-Restart Decisions

> Status: canonical checkpoint-comparison and continue-restart decision surface
> for the actual `Psion` pretraining lane, written 2026-04-02 after binding
> long-run operator posture to retained eval, backup, hardware, run-shape, and
> systems receipts.

This document records the actual-lane continue-restart decision path for
`psion_actual_pretraining_v1`.

It does not create generalized training governance. It defines one bounded
operator surface for deciding whether the latest accepted checkpoint should
continue, hold for investigation, or restart.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_continue_restart_decisions.rs`
  owns the typed checkpoint-comparison and continue-restart decision receipts.
- `crates/psionic-train/examples/psion_actual_pretraining_continue_restart_fixtures.rs`
  regenerates the committed fixtures and example run root.
- `crates/psionic-train/examples/psion_actual_pretraining_operator.rs`
  owns the actual-lane operator command that writes the retained decision.
- `scripts/train-psion-actual-pretraining.sh` exposes the canonical operator
  wrapper.
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_comparison_v1.json`
  is the committed comparison receipt fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_continue_restart_decision_v1.json`
  is the committed continue-restart decision fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_continue_restart_example/`
  contains one retained example run root for the continue branch.

## Canonical Command

```bash
./TRAIN --lane actual_pretraining decide-continue-restart --run-root <path>
```

The command reads:

- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `checkpoints/latest_accepted_checkpoint_backup_receipt.json`
- `evals/latest_checkpoint_eval_decision.json`
- or `evals/latest_checkpoint_eval_failure.json`
- `preflight/hardware_qualification.json`
- `preflight/run_shape_qualification.json`
- the committed systems bundle

It writes:

- `decisions/checkpoint_comparison_step-<optimizer_step>.json`
- `decisions/latest_checkpoint_comparison.json`
- `decisions/continue_restart_decision_step-<optimizer_step>.json`
- `decisions/latest_continue_restart_decision.json`

It also refreshes:

- `status/current_run_status.json`
- `status/retained_summary.json`
- `closeout/closeout_bundle.json`
- `dashboard/current_dashboard.json`

## Comparison Inputs

The checkpoint-comparison receipt binds one accepted checkpoint to:

- the latest retained backup receipt
- the latest retained checkpoint-eval decision or failure
- the retained hardware qualification
- the retained run-shape qualification
- the trusted-cluster throughput anchor inside the systems bundle

The current comparison rows check:

- checkpoint-eval receipt availability
- checkpoint-eval decision state
- checkpoint-eval pass rate
- checkpoint-eval aggregate score
- backup state
- hardware admission state
- run-shape admission state
- throughput ratio against the trusted-cluster anchor
- step-latency ratio against the trusted-cluster anchor
- checkpoint-write-throughput ratio against the trusted-cluster anchor
- dataloader stall count

## Continue Threshold

The actual lane now uses one explicit continue threshold:

- checkpoint-eval decision must be `continue`
- durable backup state must be `backed_up`
- hardware admission state must stay `admitted`
- run-shape admission state must stay `admitted`
- observed throughput must stay at or above `90%` of the trusted-cluster
  anchor
- observed step latency must stay at or below `115%` of the trusted-cluster
  anchor
- observed checkpoint write throughput must stay at or above `90%` of the
  trusted-cluster anchor
- dataloader stall count must stay at or below `1`

## Decision States

The retained decision state is one of:

- `continue`
- `hold_and_investigate`
- `restart_from_last_accepted_checkpoint`

The mapping is:

- `continue`
  when checkpoint eval is green and every retained health and throughput row is
  inside the continue threshold
- `hold_and_investigate`
  when checkpoint eval is missing or failed, when the eval itself requests
  review, or when backup, health, or throughput rows are outside the continue
  threshold
- `restart_from_last_accepted_checkpoint`
  when checkpoint eval explicitly lands on the restart branch and the retained
  backup plus health receipts remain green enough to trust restart from the
  latest accepted checkpoint

## Status And Closeout Effect

After the command runs, the retained status phase becomes:

- `continue_decision_recorded`
- `hold_decision_recorded`
- or `restart_decision_recorded`

The retained summary, launcher log, and provisional closeout bundle also
repeat the decision so later rehearsal and closeout work can consume one
machine-readable operator posture instead of re-deriving it from raw receipts.

## Claim Boundary

This surface now proves:

- the actual lane has one explicit machine-readable continue-restart policy
- the policy consumes retained eval, backup, hardware, and run-shape receipts
  directly
- the continue threshold is tied to the trusted-cluster systems anchor instead
  of operator intuition

It does not yet prove:

- that the operator executed the chosen continue or restart action
- completed end-to-end lane rehearsal and final closeout claims
- continuation-stage handoff proof

## Related Docs

- `docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md`
- `docs/PSION_ACTUAL_PRETRAINING_CHECKPOINT_EVALS.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`
- `docs/TRAIN_SYSTEM.md`
