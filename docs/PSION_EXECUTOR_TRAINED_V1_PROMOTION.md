# Psion Executor `trained-v1` Promotion

> Status: canonical `PSION-0807` / `#782` record, updated 2026-03-30 after
> landing the first retained executor-capable `trained-v1` promotion packet.

This document records the first valid `trained-v1` promotion packet for the
bounded executor lane.

## Canonical Fixtures

- `fixtures/psion/executor/psion_executor_trained_v1_promotion_v1.json`
- `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v1_descriptor.json`
- `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v1_artifact_manifest.json`
- `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v1_lineage_contract.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_trained_v1_promotion_fixtures
```

## What Landed

`psionic-train` now owns one typed `trained-v1` promotion packet that binds:

- the retained optimizer ablation as the non-saturated threshold-clearing
  receipt
- the retained trace-family weighting ablation as the exactness-preservation
  receipt
- the retained supervision-density ablation as the held-out and stability
  receipt
- the retained tokenizer/architecture gate as the evidence-gate receipt
- the retained long-run rehearsal packet
- the retained bounded closeout-status packet
- the retained Mac export-inspection packet
- the retained Mac -> 4080 -> Mac roundtrip packet
- the retained 4080 decision-grade visibility packet
- one new `trained-v1` descriptor, artifact-manifest, and lineage-contract set

That means the executor lane now has one promotion answer that stays inside the
frozen phase-one naming rule instead of just carrying separate ablation wins.

## Current Retained Truth

The retained digests and promotion facts are frozen in the canonical fixture.
This doc is updated alongside that fixture and the workspace roadmap.

## Honest Current Meaning

This packet does **not** widen the executor lane.

It does prove something narrower and useful:

- one first `trained-v1` candidate now exists as a retained promotion packet
- the candidate keeps `reference_linear` green as the measured truth anchor
- the candidate keeps admitted-workload `hull_cache` green as the fast-route
  target
- the candidate keeps export, local-cluster, CPU-route, and consumer-seam
  posture explicit instead of inferring them from earlier receipts

The claim boundary remains phase-one bounded:

- admitted executor workload family only
- bounded route-replacement only
- no broader executor-family widening
- tokenizer work still blocked without new evidence

## Validation

- `cargo run -q -p psionic-train --example psion_executor_trained_v1_promotion_fixtures`
- `cargo test -q -p psionic-train builtin_executor_trained_v1_promotion_packet_is_valid -- --exact --nocapture`
- `cargo test -q -p psionic-train trained_v1_promotion_fixture_matches_committed_truth -- --exact --nocapture`
