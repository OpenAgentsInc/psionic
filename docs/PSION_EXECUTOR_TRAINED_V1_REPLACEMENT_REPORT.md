# Psion Executor `trained-v1` Replacement Report

> Status: canonical `PSION-0808` / `#783` record, updated 2026-03-30 after
> landing the first retained bounded `trained-v1` replacement report.

This document records the first final replacement verdict for the admitted
executor lane after the `trained-v1` promotion packet exists.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_trained_v1_replacement_report_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_trained_v1_replacement_report_fixtures
```

## What Landed

`psionic-train` now owns one typed replacement report that binds:

- the retained `trained-v1` promotion packet
- the retained unified throughput report
- the retained long-run rehearsal packet
- the retained Mac export-inspection packet
- the retained bounded Percepta closeout-status packet

That means the executor lane now has one final retained answer for what stayed
green, what improved, and what bounded claim status is actually supported for
`trained-v1` relative to `trained-v0`.

## Current Retained Truth

The retained digests and final bounded replacement facts are frozen in the
canonical fixture.

- report digest:
  `4290e73f0a7c818e3ccaecde497ee5c1c3003827f7a31fdc60ca00ff3b9cbdff`
- promotion-packet digest:
  `30e39b71c2e3fcc9dc052bf02c12f9a178e1bcdd8fb415bc1200fae033c5cd1d`
- candidate model id:
  `tassadar-article-transformer-trace-bound-trained-v1`
- candidate route id:
  `tassadar.article_route.direct_hull_cache_runtime.v1`
- preserved gate count: `10`
- improved metric count: `7`
- outcome count: `5`
- bounded claim status: `green_bounded_replacement_ready`
- replacement decision: `publish_trained_v1_replacement_report`
- throughput outcome digest:
  `ff12ece15c7917e2c430cb139d81c36c2d9e2964f9ee8197275664314fc037a7`
- long-run rehearsal outcome digest:
  `50ebd9cad8abd98704488103686b6c21aa78f5278e85b8bb6dc19ee5666238d2`
- export outcome digest:
  `9d6a39d78400f4a0c6c86398b677b9880080e8351653b3f68ccadb6e4a06aa8a`
- bounded closeout outcome digest:
  `9856bfc3735ddc9f89a2a9ce6a49c9aea166133542237a253c8a0644b19c1185`
- `reference_linear` delta: `73679.40346165793`
- `hull_cache` delta: `224815.26180754835`
- `hull_cache` speedup improvement: `0.059403490993948926`
- CPU gap reduction: `0.06276315967336421`
- exactness net delta bps: `8`
- held-out delta bps: `1`
- training steps-per-second delta: `5.209403645610266`

## Honest Current Meaning

This report still does **not** widen the executor lane.

It does prove the phase-one bounded replacement claim more cleanly:

- every promotion gate stayed green
- the improved metric set is now frozen in one final retained report
- throughput, stability, recovery, export, and bounded closeout stay explicit
  as separate retained outcomes instead of being implied from prior receipts
- the executor lane now has one honest `trained-v0` -> `trained-v1`
  replacement verdict on the admitted workload family

The claim boundary remains narrow:

- admitted executor workload family only
- bounded route replacement only
- `reference_linear` remains the measured baseline truth anchor
- admitted-workload `hull_cache` remains the fast-route target
- no broader executor-family or tokenizer widening is implied

## Validation

- `cargo run -q -p psionic-train --example psion_executor_trained_v1_replacement_report_fixtures`
- `cargo test -q -p psionic-train trained_v1_replacement_report -- --nocapture --test-threads=1`
