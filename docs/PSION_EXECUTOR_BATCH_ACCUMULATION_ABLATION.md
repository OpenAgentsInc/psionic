# Psion Executor Batch / Accumulation Ablation

> Status: canonical `PSION-0803` / `#778` record, updated 2026-03-30 after
> landing the first same-budget batch and accumulation ablation packet for the
> executor lane.

This document records the first retained batch-size and accumulation ablation
for the admitted executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_batch_accumulation_ablation_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_batch_accumulation_ablation_fixtures
```

## What Landed

`psionic-train` now owns one typed batch/accumulation ablation packet that
binds:

- the frozen executor decision-threshold packet
- the frozen weekly review workflow packet
- the frozen unified throughput packet

The retained packet makes one narrow point explicit:

- the same-budget 4080 variant kept the effective batch fixed
- the packet logs the memory and throughput tradeoff directly instead of hiding
  it behind throughput alone
- the lane kept zero exactness, held-out, and adversarial regressions while it
  checked that tradeoff
- the result stays reviewable but remains below the frozen promotion-noise
  threshold

## Current Retained Truth

- packet digest:
  `96e495b8dad2e1563664766fd3f51e6140c6aae6ac7c349c26abeb007074a2d9`
- baseline model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- candidate model id:
  `tassadar-article-transformer-trace-bound-trained-v0-batch-ablation-candidate-v1`
- run id:
  `tailrun-home-admitted-20260329d`
- same-budget profile id:
  `local_4080_cuda_tailnet_x86_64`
- current-best row id:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- decision-threshold digest:
  `01be6e718c71781c82fb2ee7472485c8cafcfa9d2ed265c331988235edca47bc`
- review-workflow digest:
  `c11b48bb9cb4381ccba810b5c154ffad6014c3b130c539a32b43dff4298078bf`
- unified-throughput digest:
  `ff12ece15c7917e2c430cb139d81c36c2d9e2964f9ee8197275664314fc037a7`
- baseline micro batch:
  `16`
- baseline accumulation steps:
  `4`
- candidate micro batch:
  `8`
- candidate accumulation steps:
  `8`
- baseline effective batch:
  `64`
- candidate effective batch:
  `64`
- effective batch comparable:
  `true`
- baseline peak memory GiB:
  `18.6`
- candidate peak memory GiB:
  `21.3`
- peak-memory delta GiB:
  `2.7`
- baseline training steps per second:
  `82.40252049829174`
- candidate training steps per second:
  `84.981146221507`
- training steps-per-second delta:
  `2.578625723215268`
- `reference_linear` delta above baseline:
  `44907.10899165808`
- `reference_linear` minimum meaningful delta:
  `65060.100461467104`
- `hull_cache` delta above baseline:
  `128312.2016645479`
- `hull_cache` minimum meaningful delta:
  `209005.1178118226`
- minimum `hull_cache` speedup improvement:
  `0.025946490993948856`
- minimum `hull_cache` speedup floor:
  `0.05`
- maximum CPU-gap reduction:
  `0.024192159673364078`
- maximum CPU-gap improvement floor:
  `0.05`
- exactness regression count:
  `0`
- held-out regression count:
  `0`
- adversarial regression count:
  `0`
- review decision:
  `log_batch_tradeoff_keep_baseline_batching`
- promotion posture:
  `log_only_keep_baseline_batching`

## Honest Current Meaning

This does not claim a second winning executor lever.

It does claim a narrower executor-lane truth:

- the lane now has one retained same-budget batch/accumulation tradeoff packet
- the effective batch stayed fixed and the added memory cost is explicit
- the throughput lift remained too small to outrun the frozen noise band
- the lane therefore logs the tradeoff for review and keeps the baseline batch
  plan in place

## Validation

- `cargo run -q -p psionic-train --example psion_executor_batch_accumulation_ablation_fixtures`
- `cargo test -q -p psionic-train psion_executor_batch_accumulation_ablation -- --nocapture`
