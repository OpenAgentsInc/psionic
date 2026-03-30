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
  `50ea5e3fcf52d3650437ef038a31a26a0bcc96fdf619b8f566a05d9764363be3`
- current-best row id:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- dashboard digest:
  `5c24954ea04b35c07e5e08709f0eecc46a23ef7cb1e62a044dea26f809c4c4d7`
- ledger digest:
  `9b86949597220f5bb4eb80c2b313fae2416c1908771ea3ae9771ec3084d06dd3`
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

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_autoblocks_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_autoblocks -- --nocapture`
