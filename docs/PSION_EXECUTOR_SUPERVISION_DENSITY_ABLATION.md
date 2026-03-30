# Psion Executor Supervision-Density Ablation

> Status: canonical `PSION-0805` / `#780` record, updated 2026-03-30 after
> landing the first same-budget supervision-density ablation packet for the
> executor lane.

This document records the first retained supervision-density ablation for the
admitted executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_supervision_density_ablation_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_supervision_density_ablation_fixtures
```

## What Landed

`psionic-train` now owns one typed supervision-density ablation packet that
binds:

- the canonical source-family contribution report
- the unified throughput packet
- the long-run rehearsal packet
- the failure-bundle taxonomy packet
- the canonical weekly review workflow

The retained packet makes one narrow rule explicit:

- supervision-density changes do not get judged on exactness alone
- held-out, throughput, and stability stay in the same packet and same review
  path
- the retained candidate posture only survives because all four dimensions stay
  green together

## Current Retained Truth

- packet digest:
  `dc276e6cbb73657d3a94d3c6d74775f32727cbafa609b794731fbc04302dddbe`
- baseline model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- candidate model id:
  `tassadar-article-transformer-trace-bound-trained-v0-supervision-density-candidate-v1`
- run id:
  `tailrun-home-admitted-20260329f`
- same-budget profile id:
  `local_4080_cuda_tailnet_x86_64`
- source-family contribution digest:
  `124f39356d3b439af224f99e72220e67ad05b212f73433d93c8f141c3354e794`
- unified-throughput digest:
  `ff12ece15c7917e2c430cb139d81c36c2d9e2964f9ee8197275664314fc037a7`
- long-run rehearsal digest:
  `50ebd9cad8abd98704488103686b6c21aa78f5278e85b8bb6dc19ee5666238d2`
- failure-bundle taxonomy digest:
  `167de1726490b46baa6ab8dba39f1a10d19bec10180614bc2c72e515596ab0aa`
- review-workflow digest:
  `c11b48bb9cb4381ccba810b5c154ffad6014c3b130c539a32b43dff4298078bf`
- exactness delta bps:
  `6`
- held-out delta bps:
  `1`
- throughput steps-per-second delta:
  `3.312400000000011`
- stability regression count:
  `0`
- all dimensions green:
  `true`
- review decision:
  `retain_supervision_density_variant_for_candidate`
- promotion posture:
  `retain_supervision_density_variant_for_trained_v1_candidate`

## Honest Current Meaning

This does not claim that supervision density is the single decisive winning
lever.

It does claim a narrower and more useful executor-lane truth:

- the lane now has one retained same-budget supervision-density packet
- exactness, held-out, throughput, and stability are all judged together
- the retained candidate posture stays honest because none of those dimensions
  were allowed to borrow against another

## Validation

- `cargo run -q -p psionic-train --example psion_executor_supervision_density_ablation_fixtures`
- `cargo test -q -p psionic-train builtin_executor_supervision_density_ablation_packet_is_valid -- --exact --nocapture`
- `cargo test -q -p psionic-train supervision_density_ablation_fixture_matches_committed_truth -- --exact --nocapture`
