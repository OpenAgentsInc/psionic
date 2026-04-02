# Psion Actual Pretraining Run-Shape Qualification

> Status: canonical actual-lane throughput, storage, and dataloader
> qualification receipt, written 2026-04-02 after binding CS336 A2-style
> systems qualification directly into non-dry-run launcher admission.

This document freezes the machine-readable run-shape qualification surface for
the canonical actual `Psion` pretraining lane.

It does not replace later backup, auto-eval, or alert automation. It does make
one retained throughput/storage/dataloader receipt mandatory before non-dry-run
start or resume can stage the actual lane.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_run_shape_qualification.rs`
  owns the typed observation and qualification contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_run_shape_qualification_fixtures.rs`
  regenerates the committed admitted observation and qualification fixtures.
- `fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json`
  is the canonical admitted operator snapshot fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_run_shape_qualification_v1.json`
  is the canonical retained qualification receipt fixture.

Stable schema versions:

- `psion.actual_pretraining_run_shape_observation.v1`
- `psion.actual_pretraining_run_shape_qualification.v1`

## What The Receipt Freezes

The run-shape qualification receipt binds one retained preflight decision to:

- the frozen actual-lane baseline-tools bundle
- the frozen actual-lane data bundle
- the frozen actual-lane systems bundle
- the frozen actual-lane evidence contract
- one retained throughput probe
- one retained storage probe
- one retained dataloader probe
- one admitted or refused launch decision

The receipt checks the exact concerns the roadmap called out:

- throughput floor against the trusted-cluster anchor
- step-latency ceiling
- checkpoint-write throughput floor
- run-root writeability
- minimum retained storage headroom
- dataset identity and max-sequence-token match
- deterministic replay truth
- planned-horizon dataloader coverage
- bounded dataloader stall posture

## Operator Behavior

`./TRAIN --lane actual_pretraining start|resume` now always writes:

- `preflight/run_shape_qualification.json`

Behavior differs by posture:

- `--dry-run` may still succeed with a refused receipt because it is only a
  planner surface
- non-dry-run `start` or `resume` now refuse when either
  `preflight/hardware_qualification.json` or
  `preflight/run_shape_qualification.json` is not `admitted`

The operator may either:

- let the launcher probe the selected local run root and retain an honest local
  refusal-grade observation
- supply `--run-shape-observation <path>` to consume a retained admitted
  observation snapshot instead

That keeps current repo behavior honest. The launcher can measure bounded local
storage behavior itself, but it does not pretend a local dry run has already
measured real H100-cluster throughput or full-horizon dataloader replay.

## Claim Boundary

This surface honestly claims:

- actual-lane launch now consumes one retained throughput/storage/dataloader
  qualification receipt
- non-dry-run launch and resume now fail closed when real-shape systems
  qualification is missing or red
- the frozen systems, data, and baseline-tools bundles now feed one concrete
  launcher admission artifact instead of staying abstract contract truth

It does not yet claim:

- durable checkpoint backup or auto-resume closure
- automatic checkpoint eval triggering
- live dashboards or alert fan-out
- completed distributed cluster execution
