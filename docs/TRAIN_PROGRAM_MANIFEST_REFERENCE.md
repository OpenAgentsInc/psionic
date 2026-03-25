# Cross-Provider Training Program Manifest Reference

> Status: canonical `XTRAIN-1` / `#517` record, updated 2026-03-25 after
> landing the first typed cross-provider training-program manifest in
> `crates/psionic-train/src/cross_provider_training_program_manifest.rs`.

This document records the first provider-neutral training-program manifest in
Psionic.

## Canonical Runner

Run the contract checker from the repo root:

```bash
scripts/check-cross-provider-training-program-manifest.sh
```

## What Landed

`psionic-train` now owns one typed root manifest for the first cross-provider
pretraining program.

The landed surface includes:

- `CrossProviderTrainingProgramManifest`
- `CrossProviderComputeSourceClass`
- `CrossProviderExecutionClass`
- `CrossProviderTrainingProgramStageAuthority`
- `CrossProviderTrainingProgramArtifactRoots`
- `CrossProviderTrainingProgramBudgetPolicy`
- `CrossProviderTrainingProgramEvidenceAuthority`
- `write_cross_provider_training_program_manifest(...)`
- the canonical fixture
  `fixtures/training/cross_provider_training_program_manifest_v1.json`
- the checker
  `scripts/check-cross-provider-training-program-manifest.sh`

## What The Manifest Makes Explicit

The first manifest freezes these seams in one machine-legible object:

- one stable program-manifest id and program-family id
- one stable run-id template for the first cross-provider pretraining program
- one canonical stage authority bound to the Psion pretrain stage
- one canonical checkpoint family
- one canonical dataset family
- one canonical environment package key
- one exact set of admitted compute-source classes
- one exact set of admitted execution classes
- one shared artifact-root layout for launch, checkpoints, metrics,
  visualization, and final evidence
- one reserved final-evidence surface
- one budget posture for wallclock, dense ranks, validators, contributors, and
  cost

## Run-Graph Binding

The manifest is not just a detached JSON contract.

It now binds directly into `TrainingRunState` through:

- `program_manifest_id`
- `program_manifest_digest`

The manifest binding path validates:

- the run stage id
- the checkpoint family
- the environment key
- the run-id prefix implied by the manifest template

before the run graph may claim it is operating under the cross-provider program
authority.

## Pass Criteria

The contract is green only if all of the following stay true:

- the committed fixture matches the generator output exactly
- the manifest stays bound to the canonical Psion pretrain stage authority
- the admitted compute-source and execution-class sets stay explicit
- the run graph can bind the manifest id and digest into `TrainingRunState`
- the final-evidence and artifact-root surfaces remain explicit instead of
  provider-defined

## Current Limits

This issue intentionally does not claim:

- provider-neutral compute-source machine reports
- provider-neutral launch binding
- real dense distributed runtime closure
- final provider-neutral evidence-bundle schema closure
- mixed-backend dense training closure

This issue makes the root training-program authority real first.
