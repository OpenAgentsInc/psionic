# Psion Executor Baseline Truth

> Status: canonical `PSION-0106` / `#711` baseline-truth record, updated
> 2026-03-30.

## Why This Doc Exists

The frozen executor packs are only useful if the repo has one explicit
`trained-v0` baseline packet tied to those packs.

This doc explains the first committed baseline-truth record and the retained
report surfaces it reconstructs.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_baseline_truth_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_baseline_truth_fixtures
```

## What Landed

`psionic-train` now owns one typed baseline-truth record for the current
executor baseline model:

- model id: `tassadar-article-transformer-trace-bound-trained-v0`
- record id: `psion_executor_baseline_truth_v1`
- record digest:
  `1cbcce5abbae31597533a62e80d5c5e1e4aa622410b883ac5a06c02f0f264784`

The record reconstructs all eleven frozen suites across:

- `tassadar.eval.frequent.v0`
- `tassadar.eval.promotion.v0`

It also keeps the retained source-report digests explicit:

- article benchmark report digest:
  `3776767477f8e61ab1e5aaf496296f663d2f190739dce6b9235a0f61110b5606`
- article generalization gate digest:
  `a55aaf47f82bf19828185bbae659951c5d6e058324b4f9399b645f7d652dbc62`
- article evaluation-independence audit digest:
  `88ddea608361c0596f92a61ad17a899aa048ce93b9d3a12d9885d3e4055ae475`

## Current `trained-v0` Truth

The committed baseline packet is green on the first frozen executor surfaces:

- frequent exactness cases are saturated at `10000` bps
- promotion exactness cases are saturated at `10000` bps
- held-out promotion truth stays green through the retained generalization
  gate
- adversarial promotion truth stays green through the retained generalization
  gate
- held-out exclusions remain explicit as boundary rows instead of being
  flattened into one scalar
- operator review, runtime blocker, and serving blocker suites stay checklist
  backed and explicitly green

The packet also keeps the current claim boundary visible:

- `reference_linear` remains the measured baseline truth anchor
- `hull_cache` remains the admitted fast-route target on the executor family
- manual checklist suites remain manual instead of being mislabeled as pure
  automation

## Honest Boundary

This baseline packet reconstructs the frozen pack truth from the current
committed `trained-v0` report surfaces. It does not pretend the repo has
already run fresh phase-one MLX or 4080 decision-grade training jobs.

That is intentional. EPIC 1 needs one committed pack-truth record before later
variance, formatting-audit, and run-admission work can reason about deltas
honestly.
