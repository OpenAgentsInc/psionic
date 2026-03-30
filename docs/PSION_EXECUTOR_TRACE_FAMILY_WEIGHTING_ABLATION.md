# Psion Executor Trace-Family Weighting Ablation

> Status: canonical `PSION-0804` / `#779` record, updated 2026-03-30 after
> landing the first same-budget trace-family weighting ablation packet for the
> executor lane.

This document records the first retained trace-family weighting ablation for
the admitted executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_trace_family_weighting_ablation_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_trace_family_weighting_ablation_fixtures
```

## What Landed

`psionic-train` now owns one typed trace-family weighting ablation packet that
binds:

- the canonical executor mixture packet
- the canonical source-family contribution report
- the canonical misleading-win rollback-policy packet

The retained packet makes three things explicit:

- one same-budget weight shift now exists as a durable candidate packet instead
  of a review-memory note
- the packet records per-family and per-slice deltas directly
- the held-out rollback guard stayed inactive because the retained held-out
  slices improved rather than regressed

## Current Retained Truth

- packet digest:
  `1df209f442f60ee516bb0bc42afd579788dfc7f489eda39e953fe8f1f07aa44d`
- baseline mixture id:
  `psion_executor_canonical_mixture_v0`
- candidate mixture id:
  `psion_executor_canonical_mixture_trace_weighting_candidate_v1`
- baseline model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- candidate model id:
  `tassadar-article-transformer-trace-bound-trained-v0-trace-weighting-candidate-v1`
- run id:
  `tailrun-home-admitted-20260329e`
- same-budget profile id:
  `local_4080_cuda_tailnet_x86_64`
- source-family contribution digest:
  `124f39356d3b439af224f99e72220e67ad05b212f73433d93c8f141c3354e794`
- rollback-policy digest:
  `36cd968e3dbeb3810a4da9ca8ebcb1b2b097af2077993c469f461b98dceba9cf`
- changed family count:
  `3`
- exactness net delta bps:
  `8`
- held-out negative delta count:
  `0`
- adversarial negative delta count:
  `0`
- rollback applied:
  `false`
- rollback decision:
  `no_rollback_retained_trace_weight_shift`
- review decision:
  `retain_trace_weight_shift_no_rollback`
- promotion posture:
  `retain_trace_weight_variant_for_trained_v1_candidate`

## Honest Current Meaning

This does not claim a promoted mixture already exists.

It does claim a narrower and useful executor-lane truth:

- the lane now has one retained same-budget trace-family weighting candidate
- the mixture change stays inside one lever class and cites the existing
  rollback policy directly
- source-family and slice deltas are explicit instead of being folded into one
  blended review sentence
- the retained change is candidate-worthy only because held-out behavior stayed
  clean

## Validation

- `cargo run -q -p psionic-train --example psion_executor_trace_family_weighting_ablation_fixtures`
- `cargo test -q -p psionic-train builtin_executor_trace_family_weighting_ablation_packet_is_valid -- --exact --nocapture`
- `cargo test -q -p psionic-train trace_family_weighting_ablation_fixture_matches_committed_truth -- --exact --nocapture`
