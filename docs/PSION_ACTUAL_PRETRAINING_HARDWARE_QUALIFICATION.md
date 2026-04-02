# Psion Actual Pretraining Hardware Qualification

> Status: canonical actual-lane hardware admission receipt, written 2026-04-02
> after binding retained preflight evidence and fail-closed launcher gating
> into `psion_actual_pretraining_v1`.

This document freezes the machine-readable hardware qualification surface for
the canonical actual `Psion` pretraining lane.

It does not replace later live dashboards or long-run alerting. It does make
one retained preflight receipt mandatory before non-dry-run start or resume can
stage the actual lane.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_hardware_qualification.rs`
  owns the typed observation and qualification contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_hardware_qualification_fixtures.rs`
  regenerates the committed admitted observation and qualification fixtures.
- `fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json`
  is the canonical admitted operator snapshot fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_hardware_qualification_v1.json`
  is the canonical retained qualification receipt fixture.

Stable schema versions:

- `psion.actual_pretraining_hardware_observation.v1`
- `psion.actual_pretraining_hardware_qualification.v1`

## What The Receipt Freezes

The hardware qualification receipt binds one retained preflight decision to:

- the frozen actual-lane topology/storage bundle
- the frozen actual-lane systems bundle
- the frozen actual-lane evidence contract
- one observed worker inventory
- one redacted credential-source check set
- one checkpoint-restore readiness bit
- one admitted or refused launch decision

The receipt checks the exact concerns the roadmap called out:

- backend family
- worker count
- H100 inventory shape
- free-memory floor
- temperature ceiling
- ECC cleanliness
- throttling absence
- no resident compute processes
- declared storage credential presence
- checkpoint-restore readiness

## Operator Behavior

`./TRAIN --lane actual_pretraining start|resume` now always writes:

- `preflight/hardware_qualification.json`

Behavior differs by posture:

- `--dry-run` may still succeed with a refused receipt because it is only a
  planner surface
- non-dry-run `start` or `resume` now refuse when the retained hardware
  qualification receipt is not `admitted`

The operator may either:

- let the launcher probe the local host with `nvidia-smi` plus redacted
  credential-source checks
- supply `--hardware-observation <path>` to consume a retained operator
  snapshot instead

That keeps the current repo behavior honest. The actual launcher still lives in
this repo and does not yet orchestrate four remote H100 hosts itself, so the
optional observation file is the current path for binding remote cluster truth
into the retained preflight receipt.

## Claim Boundary

This surface honestly claims:

- actual-lane launch now consumes one retained hardware admission receipt
- non-dry-run launch and resume now fail closed on unhealthy hardware
- credential and checkpoint-restore readiness now sit in the same retained
  preflight receipt as GPU health

It does not yet claim:

- continuous live health monitoring during training
- automatic device replacement or topology replanning
- dashboard or alert fan-out
- post-launch checkpoint backup or auto-eval closure

The companion systems admission receipt now lives in
`docs/PSION_ACTUAL_PRETRAINING_RUN_SHAPE_QUALIFICATION.md`. Non-dry-run launch
and resume require both retained preflight receipts to be green.
