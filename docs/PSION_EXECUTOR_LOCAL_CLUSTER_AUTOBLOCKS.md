# Psion Executor Local Cluster Autoblocks

> Status: canonical `PSION-0404` / `#737` record, updated 2026-03-30 after
> landing the first local-cluster auto-block report for phase exit and
> promotion.

This document records the first canonical block surface that turns missing
executor evidence into explicit machine-readable block rows instead of review
prose and now separates phase-exit closure from promotion-only eval blockers.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_local_cluster_autoblocks_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_local_cluster_autoblocks_fixtures
```

## What Landed

`psionic-train` now owns one typed local-cluster auto-block report that binds:

- the retained local-cluster dashboard packet
- the searchable local-cluster ledger
- the frozen baseline-truth record
- the admitted 4080 frequent-eval attachment packet

The report keeps four explicit block rows:

- missing eval fact
- missing recovery fact
- missing export fact
- missing `reference_linear` baseline-anchor fact

Phase exit and promotion now resolve from that same report, but they no longer
collapse into one boolean when the roundtrip closes before the eval truth does.

## Current Retained Truth

- report digest:
  `f5e86eff4633b2710aac7c0e65ffd9a517d23be2574fe53890bcdf13ba4e8bbc`
- current-best row id:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- dashboard digest:
  `026da39b01fff5eb4e93025f0a39ad5356c4d8368e603b34b3690e16b140ee28`
- ledger digest:
  `618605effd540810a884fb6797bee683327033cdaae3e79fa5ab0fec51b7b63c`
- baseline truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- frequent-eval digest:
  `b61245ffc124b3a8fd9ec3e15bee20782fe4538f30a4595829e735bdbc360fb7`
- phase exit blocked:
  `false`
- promotion blocked:
  `true`
- active phase-exit block ids:
  none
- active promotion block ids:
  `missing_eval_fact_current_best`

## Current Gate Posture

- `missing_eval_fact_current_best`: `blocked_missing_eval_fact`
- `missing_recovery_fact_current_best`: `green`
- `missing_export_fact_current_best`: `green`
- `missing_reference_linear_anchor`: `green`

## Honest Current Meaning

The block report now says something more precise and useful:

- the current-best retained row still inherits missing or unscored
  frequent-pack coverage from the admitted 4080 frequent-eval packet
- the current-best retained row no longer has an export blocker because the
  roundtrip packet closes the Mac-side validation loop explicitly
- recovery is not the blocker
- the frozen baseline still keeps `reference_linear` visible as the measured
  truth anchor

So local-cluster phase exit is now green for real reasons, while promotion
stays blocked for real reasons. The report keeps those two gate surfaces
separate instead of forcing them into the same answer.

The follow-on unified throughput packet now extends this block discipline into
replacement review by keeping serving-throughput regression machine-readable on
top of the retained dashboard and Mac export packet:

- `docs/PSION_EXECUTOR_UNIFIED_THROUGHPUT_REPORTING.md`
- `fixtures/psion/executor/psion_executor_unified_throughput_reporting_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_autoblocks_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_autoblocks -- --nocapture`
