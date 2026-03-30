# Psion Executor Baseline

> Status: canonical `PSION-0002` / `#701` frozen executor baseline record,
> updated 2026-03-30.

## Why This Doc Exists

Every later executor-lane evaluation, export, replacement, and promotion
verdict needs one stable baseline record.

This doc freezes the current executor-capable baseline so later issues do not
silently drift on model id, route id, fast-route identity, CPU validation
matrix, or workload family.

## Canonical Baseline Identity

- baseline model id: `tassadar-article-transformer-trace-bound-trained-v0`
- baseline route id:
  `tassadar.article_route.direct_hull_cache_runtime.v1`
- baseline fast decode: `hull_cache`

These identifiers remain phase-one truth unless a later issue explicitly
replaces them through the retained executor replacement and promotion flow.

## Admitted CPU Validation Matrix

The frozen phase-one CPU validation matrix is:

- `host_cpu_aarch64`
- `host_cpu_x86_64`

No executor replacement candidate counts as phase-one ready unless it preserves
green posture on this admitted CPU matrix.

## Representative Executor Workloads

The representative phase-one executor workloads are:

- `long_loop_kernel`
- `sudoku_v0_test_a`
- `hungarian_matching`

These are the workloads that later executor promotion, route replacement, and
bounded Percepta/Tassadar-closeout work must cite by default.

The retained packet that now freezes this trio as the canonical bounded
closeout set lives at:

- `docs/PSION_EXECUTOR_ARTICLE_CLOSEOUT_SET.md`
- `fixtures/psion/executor/psion_executor_article_closeout_set_v1.json`

## Use Rule

Use this doc as the stable reference when a later issue needs to say:

- which model artifact is the current executor incumbent
- which route id is the current executor route
- which fast decode is the admitted phase-one target
- which CPU matrix remains mandatory
- which workloads are the representative closeout set before later widening
