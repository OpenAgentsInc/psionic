# Psion Actual Pretraining Scaling Bundle

> Status: canonical actual-lane scaling and budget-selection authority, written
> 2026-04-02 after freezing one bounded recipe-family comparison for
> `psion_actual_pretraining_v1`.

This document freezes the canonical scaling bundle consumed by the actual
`Psion` pretraining lane.

It does not create an open-ended scaling-law research service. It binds one
bounded CS336 A3-shaped model-size and token-budget comparison directly into
actual recipe authority so the current recipe can cite retained scaling
evidence instead of prose.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_scaling_bundle.rs` owns
  the typed scaling-bundle contract.
- `crates/psionic-train/examples/psion_actual_pretraining_scaling_bundle_fixtures.rs`
  regenerates the committed fixture from retained recipe, pretrain, benchmark,
  and observability artifacts.
- `fixtures/psion/pretrain/psion_actual_pretraining_scaling_bundle_v1.json` is
  the canonical machine-readable scaling bundle consumed by the actual lane.
- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md` freezes the recipe that cites this
  bundle.

The stable schema version is:

- `psion.actual_pretraining_scaling_bundle.v1`

## Frozen Scaling Authority

The bundle fixes:

- one bounded `64M -> 128M -> 256M` recipe-family comparison tied to the same
  admitted actual-lane data and systems surfaces
- one measured 128M anchor based on the retained broader-pretraining receipts
- one smaller eligible projection and one larger over-budget projection
- one largest-eligible selection rule for model size, token budget, and stage
  length
- one benchmark-floor and validation-loss threshold set that recipe decisions
  must satisfy

The current frozen rule keeps:

- model size anchor: `internal128m`
- admitted recipe id: `psion_actual_pretraining_recipe_v1`
- tokens-per-parameter policy: `8`
- max stage length: `4_200_000 ms`
- max total cost: `600_000_000 microusd`
- max validation loss: `1220 milli`
- min reasoning floor: `8000 bps`

That rule preserves the current 128M recipe because it is the largest candidate
that still clears the admitted stage-length, cost, and quality thresholds on
the frozen four-node H100 topology.

## Claim Boundary

This bundle is recipe-serving scaling authority for the actual lane. It is not
an unbounded sweep framework, an always-on scaling-law service, or a detached
research lane outside the frozen actual pretraining contract.
