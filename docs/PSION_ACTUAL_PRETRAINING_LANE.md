# Psion Actual Pretraining Lane

> Status: canonical actual-lane spec, written 2026-04-02 after freezing one
> repo-owned actual-lane identifier above the bounded reference pilot.

This document freezes one canonical actual `Psion` pretraining lane above the
existing bounded pilot surfaces.

It does not claim that all operator hardening is finished. It does freeze the
named lane that later launcher, recipe, checkpoint, eval, and continuation
work must target.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_lane.rs` owns the typed
  actual-lane spec.
- `crates/psionic-train/examples/psion_actual_pretraining_lane_fixtures.rs`
  regenerates the canonical lane fixture.
- `fixtures/psion/pretrain/psion_actual_pretraining_lane_spec_v1.json` is the
  canonical machine-readable actual-lane spec.
- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md` freezes the canonical recipe and
  admitted topology/storage bundle consumed by this lane.

The stable schema version is:

- `psion.actual_pretraining_lane_spec.v1`

## Canonical Lane Identity

The canonical actual lane is:

- lane id: `psion_actual_pretraining_v1`
- stage-program id: `psion_actual_pretraining_program_v1`
- training run profile: `broader_pretraining`
- run-root family: `psion_actual_pretraining_runs/<run_id>`
- evidence family: `psion.actual_pretraining.evidence.v1`

The lane is explicitly anchored to:

- `fixtures/psion/trusted_cluster/psion_trusted_cluster_run_bundle_v1.json`

That anchor keeps the actual lane tied to one admitted broader-pretraining
bundle instead of letting it drift into a vague "bigger than pilot" label.

## What Counts As The Actual Lane

The actual lane is the broader-pretraining path anchored to the existing
trusted-cluster bundle:

- one explicit `pretrain` stage receipt on the broader-pretraining profile
- one broader-pretraining observability receipt
- one admitted trusted-cluster topology contract
- one admitted checkpoint-recovery bundle for the same broader-pretraining run

This is the lane later operator work should harden.

## What Does Not Count

The actual lane is not:

- `psion_accelerated_reference_pilot`
- `psion_reference_pilot`
- the bounded Google single-node pilot surfaces
- the bounded plugin-conditioned reference surfaces

Those lanes still matter for bring-up, operator proof, bounded archive or
restore checks, and fixture truth. They are not the canonical actual-lane
identity.

## Why The Distinction Matters

Without a frozen actual-lane identifier, later work would keep attaching
operator claims to whichever bounded pilot or reference surface happened to be
nearby.

This doc prevents that drift by fixing one lane id and one admitted anchor
bundle before the launcher or hardening work starts.

`./TRAIN` still defaults to the bounded reference pilot. The actual lane is now
available explicitly through:

- `./TRAIN --lane actual_pretraining start`
- `./TRAIN --lane actual_pretraining resume --run-root <path>`
- `./TRAIN --lane actual_pretraining status --run-root <path>`

Those commands materialize the actual-lane retained evidence family under
`psion_actual_pretraining_runs/<run_id>` naming without pretending that later
hardware qualification, backup, auto-eval, or cluster execution hardening is
already finished.
