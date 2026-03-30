# Psion Executor Mixture Search Cadence

> Status: canonical `PSION-0504` / `#744` record, updated 2026-03-30 after
> landing the first weekly mixture-search cadence packet for the executor
> lane.

This document records the first machine-readable cadence packet for executor
mixture search.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_mixture_search_cadence_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mixture_search_cadence_fixtures
```

## What Landed

`psionic-train` now owns one typed cadence packet that binds:

- the canonical executor mixture
- the canonical local-cluster run-registration packet
- the canonical weekly review workflow

The packet makes three operational rules explicit:

- only one new mixture version may open per weekly review window
- registration rows now carry the active mixture version id
- before lane health clears, the active mixture may keep at most two
  concurrently registered runs

## Current Retained Truth

- packet digest:
  `f082656602c575ca4dd258ce1f38d59bacaa6ce40380b8c8e9e21a7ea7219631`
- active mixture version:
  `psion_executor_canonical_mixture_v0`
- current review window:
  `2026-W14`
- max new mixture versions per review:
  `1`
- max concurrent registered runs before lane health:
  `2`
- registered run count:
  `2`
- current lane health status:
  `promotion_blocked_current_best`
- current lane health block ids:
  `missing_eval_fact_current_best`

## Honest Current Meaning

This does not claim the lane is ready for broad parallel search.

It does make the current rule explicit:

- the active mixture version is frozen into registration truth
- the lane may not quietly open uncontrolled parallel mixture experiments
- the current weekly review still reports a promotion block on current-best,
  so the cadence remains conservative on purpose

## Validation

- `cargo run -q -p psionic-train --example psion_executor_mixture_search_cadence_fixtures`
- `cargo test -q -p psionic-train psion_executor_mixture_search_cadence -- --nocapture`
