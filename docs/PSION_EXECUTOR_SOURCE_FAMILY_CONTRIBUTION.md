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
  `124f39356d3b439af224f99e72220e67ad05b212f73433d93c8f141c3354e794`
- mixture digest:
  `09e45bad390178531cb0154756fbcd6da0a9a28e9c07c14cbe2900651095a044`
- baseline-truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- local-cluster ledger digest:
  `618605effd540810a884fb6797bee683327033cdaae3e79fa5ab0fec51b7b63c`
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
- the refreshed report now cites the same mixture-version-aware registration
  and ledger stack that the weekly cadence packet uses
- the refreshed report is now also the retained evidence source for the
  misleading-mixture rollback policy packet

## Validation

- `cargo run -q -p psionic-train --example psion_executor_source_family_contribution_fixtures`
- `cargo test -q -p psionic-train psion_executor_source_family_contribution -- --nocapture`
