# Psion Executor Scheduler Ablation

> Status: canonical `PSION-0802` / `#777` record, updated 2026-03-30 after
> landing the first same-budget scheduler and warmup ablation packet for the
> executor lane.

This document records the first retained scheduler and warmup ablation for the
admitted executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_scheduler_ablation_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_scheduler_ablation_fixtures
```

## What Landed

`psionic-train` now owns one typed scheduler-ablation packet that binds:

- the frozen executor decision-threshold packet
- the frozen weekly review workflow packet
- the frozen unified throughput packet

The retained packet makes one narrow point explicit:

- the same-budget scheduler and warmup variant is valid enough to log and
  review
- it stays directly comparable to the frozen baseline
- it does not clear the frozen noise band, so it does not become a promotion
  lever
- the lane kept zero exactness, held-out, and adversarial regressions while it
  checked that

## Current Retained Truth

- packet digest:
  `b78acc30a3c31cafd51d70c8cf2f634605f87242f872d1f9c9ab7ba42557e167`
- baseline model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- candidate model id:
  `tassadar-article-transformer-trace-bound-trained-v0-scheduler-ablation-candidate-v1`
- run id:
  `tailrun-home-admitted-20260329c`
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
- baseline scheduler id:
  `cosine_decay_warmup_500`
- candidate scheduler id:
  `cosine_decay_warmup_750`
- baseline warmup steps:
  `500`
- candidate warmup steps:
  `750`
- `reference_linear` delta above baseline:
  `31819.208603658015`
- `reference_linear` minimum meaningful delta:
  `65060.100461467104`
- `hull_cache` delta above baseline:
  `100941.76142454846`
- `hull_cache` minimum meaningful delta:
  `209005.1178118226`
- minimum `hull_cache` speedup improvement:
  `0.018903490993948946`
- minimum `hull_cache` speedup floor:
  `0.05`
- maximum CPU-gap reduction:
  `0.020690159673364406`
- maximum CPU-gap improvement floor:
  `0.05`
- baseline training steps per second:
  `82.40252049829174`
- candidate training steps per second:
  `83.744821509334`
- training steps-per-second delta:
  `1.3423010110422666`
- comparable to baseline:
  `true`
- logged and reviewed:
  `true`
- exactness regression count:
  `0`
- held-out regression count:
  `0`
- adversarial regression count:
  `0`
- review decision:
  `log_scheduler_variant_keep_baseline_scheduler`
- promotion posture:
  `log_only_keep_baseline_scheduler`

## Honest Current Meaning

This does not claim a second winning lever.

It does claim a narrower executor-lane truth:

- the scheduler and warmup change was worth running under the same-budget rule
- the result stayed clean and comparable
- the result was still too small to outrun the frozen noise band
- the lane therefore logs it for review and keeps the baseline scheduler in
  place

## Validation

- `cargo run -q -p psionic-train --example psion_executor_scheduler_ablation_fixtures`
- `cargo test -q -p psionic-train psion_executor_scheduler_ablation -- --nocapture`
