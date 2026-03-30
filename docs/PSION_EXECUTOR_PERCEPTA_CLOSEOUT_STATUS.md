# Psion Executor Percepta Closeout Status

> Status: canonical `PSION-0705` / `#774` record, updated 2026-03-30 after
> adding the first bounded Percepta / Tassadar-computation closeout-status
> packet.

This document records the explicit bounded closeout-status verdict for the
executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_percepta_closeout_status_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_percepta_closeout_status_fixtures
```

## What Landed

`psionic-train` now owns one typed bounded closeout-status packet that binds:

- workload truth on the frozen closeout trio
- fast-path truth on the retained `HullKVCache` benchmark packet
- route-replacement truth on the retained Mac export inspection and carrier-binding contract
- the bounded research-only executor-style branch

## Current Retained Truth

- packet digest:
  `9856bfc3735ddc9f89a2a9ce6a49c9aea166133542237a253c8a0644b19c1185`
- closeout-set digest:
  `2de570208df4bec06457bb0699e34f42099c9a191c4eeeb31bcd2d71b8f70734`
- trace-native metrics digest:
  `0ba90acb2a4b23c74699c55ef897eb5d9f0ef01bce05d68e03d6840538541a89`
- `HullKVCache` benchmark digest:
  `67277c9c0e8d7e9f0fe4ef3c3bf882b62258712c9eb15cd425152a8c331e6668`
- research-branch digest:
  `ce7334ffa96c452d076af8053fe149d296b4b0770e97269b835a6f927c6a31c9`
- Mac export-inspection digest:
  `9d6a39d78400f4a0c6c86398b677b9880080e8351653b3f68ccadb6e4a06aa8a`
- fast-route legitimacy summary digest:
  `3c7ae6551d9d451f815b3c97c4ee6ccb3dda186466d869ea3dc831cd5c5b47e0`
- canonical model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- canonical route id:
  `tassadar.article_route.direct_hull_cache_runtime.v1`
- bounded closeout status:
  `green_bounded`
- workload truth status:
  `green`
- fast-path truth status:
  `green`
- route-replacement truth status:
  `green`
- research-branch status:
  `research_only`
- minimum retained `HullKVCache` speedup over `reference_linear`:
  `1.690977509006051`
- maximum retained `HullKVCache` remaining gap versus CPU reference:
  `2.548278294637131`
- retained candidate row count:
  `2`
- retained route checklist row count:
  `6`
- remaining limitations:
  `arbitrary_c_or_wasm_not_claimed`,
  `research_branch_remains_research_only`,
  `trained_v1_candidate_promotion_moves_to_psion_epic_8`

## Honest Meaning

This packet records whether the executor lane is `red`, `partial`, or
`green_bounded` on the bounded Percepta / Tassadar-computation target.

`green_bounded` is the honest verdict here because:

- the frozen closeout trio stays explicit and green
- fast-path truth is green on the admitted executor family
- route-replacement continuity is green on the retained export and carrier
  surfaces
- the research branch exists, but still stays research-only and non-widening

It does not claim:

- arbitrary C execution
- arbitrary Wasm universality
- phase-one closure outside the admitted executor workload family

## Validation

- `cargo run -q -p psionic-train --example psion_executor_percepta_closeout_status_fixtures`
- `cargo test -q -p psionic-train psion_executor_percepta_closeout_status -- --nocapture`
