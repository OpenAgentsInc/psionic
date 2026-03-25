# Google Training Binder Reference

> Status: canonical `XTRAIN-11` / `#527` record, updated 2026-03-25 after
> landing the Google training binder projection in
> `crates/psionic-train/src/google_training_binder_projection.rs`.

This document records the Google-specific projection layer that binds the
current single-node and two-node swarm Google lanes to the shared
cross-provider runtime binder.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-google-training-binder-projection.sh
```

## What Landed

`psionic-train` now owns one Google projection set that freezes:

- the exact shared runtime binding used by the Google single-node lane
- the exact shared runtime binding used by the Google two-node swarm lane
- the retained Google preflight, launch, startup, and finalizer scripts
- the retained checker and evidence surfaces that must stay green under the
  shared binder

The landed surface includes:

- `GoogleTrainingBinderProjectionSet`
- `GoogleTrainingBinderLaneProjection`
- the binary `google_training_binder_projection`
- the checker `scripts/check-google-training-binder-projection.sh`
- the committed fixture `fixtures/training/google_training_binder_projection_v1.json`

## Why This Matters

The shared runtime binder is not useful if the strongest operator surfaces stay
outside it. The Google lanes are the strongest current cloud operator paths in
the repo, so they are the first lanes that now explicitly consume the shared
binder instead of acting as Google-only training entrypoints.

## Current Limits

This issue intentionally does not claim:

- dense full-model Google swarm closure
- replacement of the current Google shell/operator surfaces
- new Google resource classes beyond the current runbooks

