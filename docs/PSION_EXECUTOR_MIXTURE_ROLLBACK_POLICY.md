# Psion Executor Mixture Rollback Policy

> Status: canonical `PSION-0505` / `#745` record, updated 2026-03-30 after
> landing the first rollback policy for misleading executor mixture wins.

This document records the first retained rollback-policy packet for executor
mixture review.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_mixture_rollback_policy_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mixture_rollback_policy_fixtures
```

## What Landed

`psionic-train` now owns one typed rollback-policy packet that binds:

- the canonical executor mixture packet
- the canonical source-family contribution report
- one explicit misleading-win trigger surface
- one explicit single-lever retry constraint

The packet makes three rules durable:

- a same-budget train-looking win does not count if held-out or adversarial
  slices go negative
- rollback decisions are recorded in the weekly review workflow instead of
  getting handled informally
- after a rollback trigger, the next try may change only one lever class

## Current Retained Truth

- packet digest:
  `36cd968e3dbeb3810a4da9ca8ebcb1b2b097af2077993c469f461b98dceba9cf`
- active mixture version:
  `psion_executor_canonical_mixture_v0`
- same-budget win claim present:
  `false`
- rollback triggered:
  `false`
- rollback decision:
  `hold_no_misleading_mixture_win`
- rollback status:
  `no_misleading_win_current_week`
- trigger row count:
  `2`
- max changed levers after rollback:
  `1`
- allowed lever classes:
  `source_family_weight_bps`, `held_out_exclusion_boundary`,
  `curriculum_stage_boundary`

## Honest Current Meaning

The current retained week does not show a misleading win yet.

That is still useful because the lane now has one durable policy for what
happens when that case arrives:

- train-looking exactness gains do not outrank held-out or adversarial damage
- the rollback decision is retained as review truth instead of operator memory
- the next try is constrained to one lever so the lane does not hide a bad
  week inside a many-change retry

The refreshed weekly review workflow now cites this packet directly:

- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_REVIEW_WORKFLOW.md`
- `fixtures/psion/executor/psion_executor_local_cluster_review_workflow_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_mixture_rollback_policy_fixtures`
- `cargo test -q -p psionic-train psion_executor_mixture_rollback_policy -- --nocapture`
