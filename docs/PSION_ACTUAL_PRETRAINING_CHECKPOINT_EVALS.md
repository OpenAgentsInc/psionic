# Psion Actual Pretraining Checkpoint Evals

> Status: canonical automatic checkpoint-eval surface for the actual `Psion`
> pretraining lane, written 2026-04-02 after binding saved checkpoints to one
> frozen eval pack, one retained latest-decision path, and one retained
> failure-plus-alert path.

This document records the automatic checkpoint-eval surface for
`psion_actual_pretraining_v1`.

It does not create a detached eval program. It binds one accepted checkpoint to
one frozen benchmark pack and one retained decision receipt that later
continue-vs-restart logic can consume directly.

## Canonical Artifacts

- `crates/psionic-eval/src/psion_actual_pretraining_checkpoint_eval_pack.rs`
  owns the frozen benchmark pack.
- `crates/psionic-train/src/psion_actual_pretraining_checkpoint_evals.rs`
  owns the retained decision, failure, and redacted-alert receipts.
- `crates/psionic-train/examples/psion_actual_pretraining_checkpoint_eval_fixtures.rs`
  regenerates the committed fixtures and example run roots.
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_eval_benchmark_pack_v1.json`
  is the committed benchmark pack.
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_eval_decision_v1.json`
  is the committed green decision receipt.
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_eval_failure_worker_unavailable_v1.json`
  is the committed unavailable-worker failure receipt.
- `fixtures/psion/pretrain/psion_actual_pretraining_redacted_alert_v1.json`
  is the committed redacted retry alert.
- `fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_eval_example/`
  contains success and unavailable-worker run-root examples.

## Frozen Eval Pack

The actual-lane checkpoint eval now uses one benchmark package:

- benchmark ref:
  `benchmark://psion/actual_pretraining/checkpoint_eval`
- version: `2026.04.02`
- storage key:
  `benchmark://psion/actual_pretraining/checkpoint_eval@2026.04.02`

The pack keeps the actual lane bound to the same four frozen benchmark
families already attached to the actual-lane data bundle:

- `architecture_reasoning`
- `normative_spec_reading`
- `engineering_spec_interpretation`
- `memorization_versus_reasoning`

That keeps checkpoint review tied to the current actual recipe and data
authority instead of creating a separate curriculum-only eval lane.

## Retained Paths

When `./TRAIN --lane actual_pretraining record-checkpoint ...` succeeds, it now
writes:

- `evals/checkpoint_eval_step-<optimizer_step>.json`
- `evals/latest_checkpoint_eval_decision.json`

If the automatic eval worker is unavailable, it writes:

- `evals/checkpoint_eval_failure_step-<optimizer_step>.json`
- `evals/latest_checkpoint_eval_failure.json`
- `alerts/latest_redacted_alert.json`

The current status command also prints the latest eval decision or failure and
latest alert when those retained files exist.

## Decision Output

The retained checkpoint-eval decision carries:

- exact checkpoint identity and checkpoint-manifest ref
- exact git ref, commit SHA, and dirty-tree posture
- the committed benchmark-pack fixture ref plus storage key
- four metric-gate rows, one per frozen benchmark family
- aggregate pass rate and aggregate score
- one retained decision state:
  `continue`, `hold_and_review`, or
  `restart_from_last_accepted_checkpoint`

The current committed green fixture lands with:

- aggregate pass rate: `10000`
- aggregate score: `8532`
- decision state: `continue`

## Failure And Alert Output

The retained failure receipt keeps:

- exact checkpoint identity
- exact git provenance
- the committed benchmark-pack fixture ref plus storage key
- failure kind
- retry posture
- retry delay
- the canonical alert path

The current committed failure drill uses:

- failure kind: `eval_worker_unavailable`
- resolution state: `retry_required`
- retry delay: `300` seconds

The redacted alert keeps only the retained failure path, alert kind, severity,
and redaction policy. It does not copy raw SSH targets, bucket credentials, or
service-account material into retained artifacts.

## Claim Boundary

This surface now proves:

- saved checkpoints automatically emit retained checkpoint-eval evidence on the
  actual lane
- the actual lane has one frozen checkpoint-eval benchmark pack
- unavailable eval workers retain retry-required failure evidence and a
  redacted alert instead of silently skipping eval

It does not yet prove:

- live dashboard fan-out
- alert delivery beyond the retained alert artifact
- the final continue-vs-restart policy packet
- completed distributed broader-pretraining execution

## Related Docs

- `docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`
- `docs/PSION_ACTUAL_PRETRAINING_CHECKPOINT_RECOVERY.md`
- `docs/TRAIN_SYSTEM.md`
