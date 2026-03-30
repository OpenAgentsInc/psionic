# Psion Executor Canonical Mixture V0

> Status: canonical `PSION-0501` / `#741` record, updated 2026-03-30 after
> publishing the first explicit executor-lane mixture manifest.

This document records the first canonical executor-lane mixture manifest.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_canonical_mixture_v0.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_canonical_mixture_fixtures
```

## What Landed

`psionic-train` now owns one typed executor mixture packet that fixes:

- the canonical executor corpus id
- the admitted executor task-family id
- the `MAX_TARGET_WINDOW_TOKENS=32` stage anchor
- the frozen frequent and promotion pack ids
- the initial source-family weights
- the seed-suite rows
- the held-out exclusions
- the evaluation exclusions

That means later curriculum, contribution, cadence, and rollback work now
references one stable mixture id instead of ad hoc prose.

## Current Retained Truth

- packet digest:
  `09e45bad390178531cb0154756fbcd6da0a9a28e9c07c14cbe2900651095a044`
- mixture id:
  `psion_executor_canonical_mixture_v0`
- corpus id:
  `tassadar.executor.local_cluster.canonical_mixture_v0`
- model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- task family id:
  `tassadar.executor.admitted_workload_family.v0`
- stage anchor id:
  `executor_boundary_anchor_32.v0`
- `MAX_TARGET_WINDOW_TOKENS`:
  `32`
- frozen pack ids:
  `tassadar.eval.frequent.v0`, `tassadar.eval.promotion.v0`
- held-out exclusion ids:
  `frequent_held_out_exclusions_v0`,
  `promotion_held_out_suite_v0`,
  `promotion_adversarial_suite_v0`
- initial source-family weights sum:
  `10000` bps

## Initial Source-Family Program

The first canonical mixture keeps six explicit source-family rows:

- `executor.boundary_prefix_traces`
- `executor.article_route_direct_traces`
- `executor.long_loop_kernel_traces`
- `executor.sudoku_v0_traces`
- `executor.hungarian_matching_traces`
- `executor.refusal_negative_traces`

This is intentionally narrow. The manifest is for the admitted executor lane,
not for a broad generic `Psion` family.

## Honest Current Meaning

This does not claim that the mixture is already optimized.

It does claim that:

- the executor lane now has one explicit mixture id
- later mixture search must stay tied to frozen-pack review
- held-out and evaluation exclusions are explicit before any train-looking
  mixture win is discussed

## Validation

- `cargo run -q -p psionic-train --example psion_executor_canonical_mixture_fixtures`
- `cargo test -q -p psionic-train psion_executor_canonical_mixture -- --nocapture`
