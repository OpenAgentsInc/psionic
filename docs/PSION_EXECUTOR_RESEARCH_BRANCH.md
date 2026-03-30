# Psion Executor Research Branch

> Status: canonical `PSION-0704` / `#773` record, updated 2026-03-30 after
> adding the first bounded executor-style research-branch packet.

This document records the retained research-only branch for executor-style
fast-path experiments on the bounded article closeout lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_research_branch_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_research_branch_fixtures
```

## What Landed

`psionic-train` now owns one typed bounded research-branch packet that binds:

- the frozen bounded article closeout trio
- the retained `HullKVCache` benchmark truth
- the retained Mac export inspection and replacement-publication continuity
- the post-article fast-route legitimacy and carrier-binding contract

The branch stays research-only. It does not create direct replacement
authority, and it does not widen the executor claim boundary.

## Current Retained Truth

- packet digest:
  `ce7334ffa96c452d076af8053fe149d296b4b0770e97269b835a6f927c6a31c9`
- branch id:
  `psion.executor.research_branch.executor_style.v1`
- branch status:
  `research_only`
- canonical model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- canonical route id:
  `tassadar.article_route.direct_hull_cache_runtime.v1`
- closeout-set digest:
  `2de570208df4bec06457bb0699e34f42099c9a191c4eeeb31bcd2d71b8f70734`
- `HullKVCache` benchmark digest:
  `67277c9c0e8d7e9f0fe4ef3c3bf882b62258712c9eb15cd425152a8c331e6668`
- Mac export-inspection digest:
  `9d6a39d78400f4a0c6c86398b677b9880080e8351653b3f68ccadb6e4a06aa8a`
- fast-route legitimacy summary digest:
  `3c7ae6551d9d451f815b3c97c4ee6ccb3dda186466d869ea3dc831cd5c5b47e0`
- closeout evaluation complete:
  `true`
- route-truth guard green:
  `true`
- export-truth guard green:
  `true`
- carrier-binding guard green:
  `true`
- widening-claim blocked:
  `true`
- retained experiment row digests:
  `two_d_head_hard_max_candidate`
  `bc89f6529725cd8ff3a20954c6e2578bd58d29d805ce077ba142b745fa4a21f9`
  and `executor_style_hierarchical_hull_candidate`
  `b3d694890a9d89a0dd93214af917a945b2e37c610862c173c1f698704fb455df`

## Honest Meaning

This branch exists to keep 2D-head and executor-style fast-path experiments
inside the admitted executor lane rather than letting them turn into detached
novelty.

It keeps the bounded rule explicit:

- every retained experiment stays measured on the frozen closeout trio
- `reference_linear` and the retained `HullKVCache` carrier stay the route
  truth guards
- export inspection and replacement publication stay mandatory before any later
  direct-carrier claim
- research candidates remain quarantined from direct replacement authority

It does not claim:

- arbitrary-program closure
- direct replacement authority for research candidates
- permission to bypass export, route, or carrier-binding truth

## Validation

- `cargo run -q -p psionic-train --example psion_executor_research_branch_fixtures`
- `cargo test -q -p psionic-train psion_executor_research_branch -- --nocapture`
