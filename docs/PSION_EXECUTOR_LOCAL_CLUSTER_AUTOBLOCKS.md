# Psion Executor Local Cluster Autoblocks

> Status: canonical `PSION-0404` / `#737` record, updated 2026-03-30 after
> landing the first local-cluster auto-block report for phase exit and
> promotion.

This document records the first canonical block surface that turns missing
executor evidence into explicit machine-readable block rows instead of review
prose.

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

Both phase exit and promotion now resolve from that same report instead of
relying on manual review memory.

## Current Retained Truth

- report digest:
  `1ea4d6cb038b59550548308513c63727a67f6e82b3f6e59f3573946eaf58d88f`
- current-best row id:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- dashboard digest:
  `06d6855974f0f6c5453a874f113e4b521da1ee8967f6a624de99f57e5de24a8f`
- ledger digest:
  `1650d362d9ea49099aaad6dc94459eb5530e5f35b19e74492ec47f9b5be0f632`
- baseline truth digest:
  `43b7a73e3ebdd17c9aeb692f71c0f261da409f65dded37760a4037226645a45c`
- frequent-eval digest:
  `b61245ffc124b3a8fd9ec3e15bee20782fe4538f30a4595829e735bdbc360fb7`
- phase exit blocked:
  `true`
- promotion blocked:
  `true`
- active phase-exit block ids:
  `missing_eval_fact_current_best`,
  `missing_export_fact_current_best`
- active promotion block ids:
  `missing_eval_fact_current_best`,
  `missing_export_fact_current_best`

## Current Gate Posture

- `missing_eval_fact_current_best`: `blocked_missing_eval_fact`
- `missing_recovery_fact_current_best`: `green`
- `missing_export_fact_current_best`: `blocked_missing_export_fact`
- `missing_reference_linear_anchor`: `green`

## Honest Current Meaning

The block report says something simple and useful:

- the current-best retained row still inherits missing or unscored
  frequent-pack coverage from the admitted 4080 frequent-eval packet
- the current-best retained row still has export posture
  `pending_mac_roundtrip_validation`
- recovery is not the blocker
- the frozen baseline still keeps `reference_linear` visible as the measured
  truth anchor

So local-cluster phase exit and promotion are both blocked for real reasons,
not because the repo lacks a place to say why.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_local_cluster_autoblocks_fixtures`
- `cargo test -q -p psionic-train psion_executor_local_cluster_autoblocks -- --nocapture`
