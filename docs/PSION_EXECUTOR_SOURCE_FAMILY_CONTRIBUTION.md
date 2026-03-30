# Psion Executor Source-Family Contribution Reporting

> Status: canonical `PSION-0503` / `#743` record, updated 2026-03-30 after
> landing the first source-family contribution report for the executor lane.

This document records the first canonical source-family contribution report for
executor mixture review.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_source_family_contribution_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_source_family_contribution_fixtures
```

## What Landed

`psionic-train` now owns one typed contribution report tied directly to:

- the canonical executor mixture packet
- the canonical baseline-truth packet
- the canonical local-cluster ledger

The report makes two things explicit instead of mixing them together in prose:

- per-source-family slice deltas for exactness, held-out, and adversarial
  review
- run-level throughput and stability regressions on the current-best 4080 row

## Current Retained Truth

- report digest:
  `c1e57dfd5187b778d10604f460553913786be9c5752e785ce0842f2aff66e42d`
- mixture digest:
  `09e45bad390178531cb0154756fbcd6da0a9a28e9c07c14cbe2900651095a044`
- baseline-truth digest:
  `1cbcce5abbae31597533a62e80d5c5e1e4aa622410b883ac5a06c02f0f264784`
- local-cluster ledger digest:
  `9b86949597220f5bb4eb80c2b313fae2416c1908771ea3ae9771ec3084d06dd3`
- source-family count:
  `6`
- throughput regression count:
  `3`
- stability regression count:
  `4`
- baseline row:
  `psion_executor_local_cluster_ledger_row_mlx_v1`
- candidate row:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- retained step-throughput delta:
  `-80.12809003801185`
- retained sample-throughput delta:
  `-10256.395524865517`
- retained source-token-throughput delta:
  `-2292304.3998074424`

## Honest Current Meaning

This report does not claim mixture gains yet.

It does make the current review boundary honest:

- source-family exactness, held-out, and adversarial deltas stay flat on the
  retained baseline packet because no new mixture candidate has run yet
- throughput regressions remain visible on the current-best 4080 row
- stability regressions remain visible on the current-best 4080 row
- weekly mixture review can now separate “family contributed nothing yet” from
  “hardware and recovery still have explicit debt”

## Validation

- `cargo run -q -p psionic-train --example psion_executor_source_family_contribution_fixtures`
- `cargo test -q -p psionic-train psion_executor_source_family_contribution -- --nocapture`
