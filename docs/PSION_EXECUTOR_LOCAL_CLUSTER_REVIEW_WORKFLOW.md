# Psion Executor Local Cluster Review Workflow

> Status: canonical `PSION-0405` / `#738` record, updated 2026-03-30 after
> locking the first weekly baseline and ablation review workflow for the
> admitted executor local-cluster lane.

This document records the first canonical review workflow that turns the
executor lane's weekly baseline and ablation cadence into retained machine
truth instead of review prose.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_local_cluster_review_workflow_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_local_cluster_review_workflow_fixtures
```

## What Landed

`psionic-train` now owns one typed local-cluster review workflow packet that
binds:

- the named weekly ownership cadence
- the canonical local-cluster dashboard packet
- the searchable local-cluster ledger
- the canonical auto-block report
- one recurring baseline-review template
- one recurring ablation-review template
- the current retained weekly review decisions

That means the executor lane now has one durable answer to:

- who reviews the frozen baseline every week
- who reviews same-budget ablations every week
- what evidence counts
- what evidence does not count
- which ledger row and block ids each current decision cites

## Current Retained Truth

- workflow digest:
  `920d5cc2b4a4e20bd951c4a2cc0776d3a56ad7dc3191e3aed3baab9ab95a014a`
- ownership ref:
  `docs/PSION_EXECUTOR_OWNERSHIP.md`
- dashboard digest:
  `06d6855974f0f6c5453a874f113e4b521da1ee8967f6a624de99f57e5de24a8f`
- ledger digest:
  `1650d362d9ea49099aaad6dc94459eb5530e5f35b19e74492ec47f9b5be0f632`
- baseline-truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- auto-block report digest:
  `1ea4d6cb038b59550548308513c63727a67f6e82b3f6e59f3573946eaf58d88f`
- baseline template digest:
  `042b45e975336f4f0fc5af612c8039a206c52933ff3fca645fdee442c96dfdbc`
- ablation template digest:
  `b20d44cbffa4aa65c863440dc9d74410cb14fbd3710fdb7230eaa7fe8df967d8`
- baseline decision id:
  `psion_executor_weekly_baseline_review_2026w14_v1`
- baseline decision:
  `hold_frozen_baseline`
- baseline status:
  `blocked_current_best`
- ablation decision id:
  `psion_executor_weekly_ablation_review_2026w14_v1`
- ablation decision:
  `hold_same_budget_follow_on`
- ablation status:
  `blocked_current_best`
- cited current-best row:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- cited active block ids:
  `missing_eval_fact_current_best`,
  `missing_export_fact_current_best`

## Frozen-Pack Rule

The review workflow makes one policy explicit:

- only frozen executor pack ids count as weekly review truth
- retained ledger rows, dashboard facts, and auto-block ids count as review
  evidence
- partial probes, convenience subsets, and ad hoc experiment summaries do not
  count

This keeps weekly review from mutating into an informal side channel.

## Honest Current Meaning

The workflow currently does not celebrate a new winner.

It does something more useful:

- the baseline review explicitly holds the frozen baseline because the
  current-best row still carries active block ids
- the ablation review explicitly refuses to bless a same-budget follow-on while
  those same block ids remain open
- both decisions now cite machine-readable ledger and block facts instead of
  operator memory

That is the first point of the workflow: review cadence is now retained even
when the right answer is "hold."

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_review_workflow_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_review_workflow -- --nocapture`
