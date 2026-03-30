# Psion Executor Curriculum Boundaries

> Status: canonical `PSION-0502` / `#742` record, updated 2026-03-30 after
> publishing the first stagewise executor-lane curriculum packet.

This document records the first canonical curriculum-boundary packet for the
executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_curriculum_boundaries_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_curriculum_boundaries_fixtures
```

## What Landed

`psionic-train` now owns one typed curriculum packet that binds the canonical
executor mixture to explicit stage transitions.

The packet makes three things machine-legible:

- the stage order
- the target-window boundary for each stage
- the certified frequent or promotion suites required before a stage may
  advance

## Current Retained Truth

- packet digest:
  `e237098c092ae07b637c48a1600b09fa00fb8cf57c531a1828ad27f3bec0c48e`
- curriculum id:
  `psion_executor_curriculum_boundaries_v1`
- mixture ref:
  `fixtures/psion/executor/psion_executor_canonical_mixture_v0.json`
- stage count:
  `3`
- first stage target window:
  `32`
- terminal stage certified packs:
  `tassadar.eval.frequent.v0`, `tassadar.eval.promotion.v0`

## Canonical Stages

- `boundary_anchor_32`
  - target window `32`
  - advances only after repeated retained green frequent exactness and held-out
    exclusion truth
- `frequent_pack_certification`
  - target window `128`
  - advances only after the full frequent pack stays green, including operator
    review and throughput blocker suites
- `promotion_pack_certification`
  - target window `512`
  - terminal stage tied to zero-regression promotion truth plus the retained
    runtime, serving, `reference_linear`, and `hull_cache` checks

## Honest Current Meaning

This does not make the curriculum adaptive.

It does make the stage transitions explicit and performance-driven. Loss-only
improvement is not enough to advance the executor lane once this packet exists.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_curriculum_boundaries_fixtures`
- `cargo test -q -p psionic-train psion_executor_curriculum_boundaries -- --nocapture`
