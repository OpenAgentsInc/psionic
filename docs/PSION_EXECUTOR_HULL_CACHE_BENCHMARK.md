# Psion Executor Hull Cache Benchmark

> Status: canonical `PSION-0703` / `#772` record, updated 2026-03-30 after
> adding the first explicit `HullKVCache` versus `reference_linear` benchmark
> packet for the bounded article closeout trio.

This document records the first retained fast-route benchmark packet for the
executor lane on the frozen bounded article workloads.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_hull_cache_benchmark_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_hull_cache_benchmark_fixtures
```

## What Landed

`psionic-train` now owns one typed fast-route benchmark packet that sits on top
of the retained trace-native metrics packet and makes three things explicit:

- `reference_linear` throughput on the frozen closeout trio
- `HullKVCache` throughput on the same workloads
- the promotion-block rule when fast-route serving truth turns red

## Current Retained Truth

- packet digest:
  `67277c9c0e8d7e9f0fe4ef3c3bf882b62258712c9eb15cd425152a8c331e6668`
- trace-native metrics digest:
  `0ba90acb2a4b23c74699c55ef897eb5d9f0ef01bce05d68e03d6840538541a89`
- minimum `HullKVCache` speedup over `reference_linear`:
  `1.690977509006051`
- maximum `HullKVCache` speedup over `reference_linear`:
  `3258.209807023449`
- maximum remaining gap versus CPU reference:
  `2.548278294637131`
- all serving truth green:
  `true`
- promotion blocked:
  `false`
- promotion block reason:
  `none_all_closeout_workloads_green`
- MLX candidate row digest:
  `a172525943ba2aa4d091519b76ce32ced5064a6b26841874a1f880fd5d6eebf3`
- 4080 current-best row digest:
  `7fe9c03c3c4ffbc051cb46d17237b131afbb704b021d45743bd62e034d834c9d`

## Honest Meaning

This packet does not claim that `HullKVCache` is the truth route everywhere.

It keeps the actual bounded rule explicit:

- `reference_linear` remains the measured baseline truth anchor
- `HullKVCache` is the fast-route candidate on the admitted closeout workloads
- any serving-truth regression on the bounded closeout trio blocks promotion

## Follow-On Surfaces

The next executor closeout surfaces build directly on this packet:

- `docs/PSION_EXECUTOR_TRACE_NATIVE_METRICS.md`
- `docs/PSION_EXECUTOR_ARTICLE_CLOSEOUT_SET.md`
- `docs/ROADMAP_PSION.md`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_hull_cache_benchmark_fixtures`
- `cargo test -q -p psionic-train --lib psion_executor_hull_cache_benchmark -- --nocapture`
