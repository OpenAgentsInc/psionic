# Training Execution Evidence Reference

> Status: canonical `XTRAIN-9` / `#525` record, updated 2026-03-25 after
> landing the first provider-neutral final evidence bundle family in
> `crates/psionic-train/src/training_execution_evidence_bundle.rs`.

This document records the first shared final-evidence schema family across the
current training execution classes.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-training-execution-evidence-bundle.sh
```

## What Landed

`psionic-train` now owns one final evidence bundle family with:

- one shared `TrainingExecutionEvidenceBundle`
- per-segment proof for single-node, dense-distributed, contributor-window,
  validator-only, and hybrid runs
- explicit launch, runtime, checkpoint, metric, visualization, validator, and
  after-action references
- explicit successful, degraded-success, refused, and failed dispositions
- the generator binary `training_execution_evidence_bundle`
- the checker `scripts/check-training-execution-evidence-bundle.sh`
- the fixture `fixtures/training/provider_neutral_training_execution_evidence_bundle_v1.json`

## Current Honest Boundary

This issue closes the schema family, not the entire runtime stack.

It proves:

- finalizers can emit one bundle family across differing execution classes
- refusal and degraded-success posture stay explicit
- hybrid runs can carry multiple execution classes without a hybrid-only proof
  family

It does not prove:

- that every segment in the canonical example has already happened in one real
  production run
- app rendering behavior
- mixed-backend dense portability
