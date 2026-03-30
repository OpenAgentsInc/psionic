# Psion Executor Trace-Native Metrics

> Status: canonical `PSION-0702` / `#771` record, updated 2026-03-30 after
> adding executor-trace-native metrics to the canonical ledger surface for the
> bounded article closeout lane.

This document records the first retained packet that binds the frozen bounded
article closeout set to the executor ledger using trace-native metrics rather
than generic training-only summaries.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_trace_native_metrics_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_trace_native_metrics_fixtures
```

## What Landed

`psionic-train` now owns one typed trace-native metrics packet that binds:

- the retained local-cluster ledger
- the frozen bounded article closeout set
- the retained article benchmark report for the same executor model

Each retained candidate row now keeps, per closeout workload:

- trace step count
- final-output exactness
- step exactness
- halt exactness
- trace-digest equality status
- `reference_linear` throughput
- `hull_cache` throughput
- `hull_cache` speedup over `reference_linear`
- `hull_cache` remaining gap versus CPU reference

The current ledger still has two retained candidate rows that both point at the
same executor model id. This packet therefore keeps the workload metrics
visible per candidate surface without pretending the benchmark report is a
profile-specific training-throughput measurement.

## Current Retained Truth

- packet digest:
  `0ba90acb2a4b23c74699c55ef897eb5d9f0ef01bce05d68e03d6840538541a89`
- ledger digest:
  `618605effd540810a884fb6797bee683327033cdaae3e79fa5ab0fec51b7b63c`
- closeout-set digest:
  `2de570208df4bec06457bb0699e34f42099c9a191c4eeeb31bcd2d71b8f70734`
- benchmark report SHA256:
  `f371e3cd3e065e4efecd7217f907224a054b9f42fa2ab1cb7ea3644956f93674`
- benchmark summary digest:
  `811f6ef1453f72bf2b7311a8645c7367375f185d5d39842ff23b5265e0cb5c63`
- MLX candidate row digest:
  `b779e192a1d6f538949a547024765f146f6f7d8b56b3424b5a4b88b62549bf6f`
- 4080 current-best row digest:
  `d662743f86a001520e2d6490f94a7eb44dc14d1bae384fe71aafb426ba717391`

## Honest Meaning

This packet does not widen the executor claim boundary.

It does something narrower and more useful:

- the ledger can now talk about bounded article closeout in executor-native
  terms
- the workload trio from `PSION-0701` is now visible per retained candidate
  row instead of staying implicit in a separate benchmark report
- later fast-route benchmark and bounded closeout reporting can cite one
  retained packet instead of reconstructing trace-length and exactness facts by
  hand

## Follow-On Surfaces

The next executor closeout surfaces build directly on this packet:

- `docs/PSION_EXECUTOR_PERCEPTA_CLOSEOUT_STATUS.md`
- `docs/PSION_EXECUTOR_HULL_CACHE_BENCHMARK.md`
- `docs/PSION_EXECUTOR_ARTICLE_CLOSEOUT_SET.md`
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_LEDGER.md`
- `docs/PSION_EXECUTOR_LOCAL_CLUSTER_DASHBOARD.md`
- `docs/ROADMAP_PSION.md`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_trace_native_metrics_fixtures`
- `cargo test -q -p psionic-train psion_executor_trace_native_metrics -- --nocapture`
