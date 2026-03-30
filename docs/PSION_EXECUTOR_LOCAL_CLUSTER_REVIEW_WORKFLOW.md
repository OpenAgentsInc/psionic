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
  `5b70189bcb3929f68a605307c7afbe330dd9bb8e03db02def44d0d625bb0a7bf`
- ownership ref:
  `docs/PSION_EXECUTOR_OWNERSHIP.md`
- dashboard digest:
  `5c24954ea04b35c07e5e08709f0eecc46a23ef7cb1e62a044dea26f809c4c4d7`
- ledger digest:
  `9b86949597220f5bb4eb80c2b313fae2416c1908771ea3ae9771ec3084d06dd3`
- baseline-truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- auto-block report digest:
  `50ea5e3fcf52d3650437ef038a31a26a0bcc96fdf619b8f566a05d9764363be3`
- baseline template digest:
  `2e93b06b083329e0a39a2fa2db8ed36e97347ebce4c7149f8d335bdda0ccd363`
- ablation template digest:
  `0364307416eed3d1236d8f1c64da4f36611f6335e375a9ba29b2ff7f11e4bc39`
- baseline decision id:
  `psion_executor_weekly_baseline_review_2026w14_v1`
- baseline decision:
  `hold_frozen_baseline`
- baseline status:
  `promotion_blocked_current_best`
- ablation decision id:
  `psion_executor_weekly_ablation_review_2026w14_v1`
- ablation decision:
  `hold_same_budget_follow_on`
- ablation status:
  `promotion_blocked_current_best`
- cited current-best row:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- cited active block ids:
  `missing_eval_fact_current_best`

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
  current-best row still carries an active promotion block id
- the ablation review explicitly refuses to bless a same-budget follow-on while
  that same promotion block id remains open
- both decisions now cite machine-readable ledger and block facts instead of
  operator memory

That is the first point of the workflow: review cadence is now retained even
when phase exit is green and the right promotion answer is still "hold."

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_review_workflow_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_review_workflow -- --nocapture`
