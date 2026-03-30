# Psion Executor Local Cluster Review Workflow

> Status: canonical `PSION-0405` / `#738` record, updated 2026-03-30 after
> locking the first weekly baseline, ablation, and mixture-rollback review
> workflow for the admitted executor local-cluster lane.

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
- one recurring mixture-rollback review template
- the current retained weekly review decisions

That means the executor lane now has one durable answer to:

- who reviews the frozen baseline every week
- who reviews same-budget ablations every week
- who reviews misleading mixture wins every week
- what evidence counts
- what evidence does not count
- which ledger row and block ids each current decision cites

## Current Retained Truth

- workflow digest:
  `c11b48bb9cb4381ccba810b5c154ffad6014c3b130c539a32b43dff4298078bf`
- ownership ref:
  `docs/PSION_EXECUTOR_OWNERSHIP.md`
- dashboard digest:
  `026da39b01fff5eb4e93025f0a39ad5356c4d8368e603b34b3690e16b140ee28`
- ledger digest:
  `618605effd540810a884fb6797bee683327033cdaae3e79fa5ab0fec51b7b63c`
- baseline-truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- auto-block report digest:
  `f5e86eff4633b2710aac7c0e65ffd9a517d23be2574fe53890bcdf13ba4e8bbc`
- mixture rollback-policy digest:
  `36cd968e3dbeb3810a4da9ca8ebcb1b2b097af2077993c469f461b98dceba9cf`
- baseline template digest:
  `2e93b06b083329e0a39a2fa2db8ed36e97347ebce4c7149f8d335bdda0ccd363`
- ablation template digest:
  `0364307416eed3d1236d8f1c64da4f36611f6335e375a9ba29b2ff7f11e4bc39`
- mixture-rollback template digest:
  `cbfad6ab745df942e02615f88393543d0497bde97ac47b4ce24b9f6eca81cf76`
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
- mixture-rollback decision id:
  `psion_executor_weekly_mixture_rollback_review_2026w14_v1`
- mixture-rollback decision:
  `hold_no_misleading_mixture_win`
- mixture-rollback status:
  `no_misleading_win_current_week`
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
- the mixture-rollback review now records that no misleading same-budget
  mixture win exists in the retained week, while freezing the rollback trigger
  and single-lever retry rule for future weeks
- both decisions now cite machine-readable ledger and block facts instead of
  operator memory

That is the first point of the workflow: review cadence is now retained even
when phase exit is green and the right promotion answer is still "hold."

The follow-on weekly mixture-search cadence packet now lives at:

- `docs/PSION_EXECUTOR_MIXTURE_SEARCH_CADENCE.md`
- `fixtures/psion/executor/psion_executor_mixture_search_cadence_v1.json`

The new rollback-policy packet now lives at:

- `docs/PSION_EXECUTOR_MIXTURE_ROLLBACK_POLICY.md`
- `fixtures/psion/executor/psion_executor_mixture_rollback_policy_v1.json`

The phase-two incident policy that now governs continue-vs-restart handling
lives at:

- `docs/PSION_EXECUTOR_CONTINUE_RESTART_POLICY.md`
- `fixtures/psion/executor/psion_executor_continue_restart_policy_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_review_workflow_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_review_workflow -- --nocapture`
