# Psion Executor Optimizer Ablation

> Status: canonical `PSION-0801` / `#776` record, updated 2026-03-30 after
> landing the first same-budget optimizer ablation packet for the executor
> lane.

This document records the first retained optimizer ablation for the admitted
executor lane.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_optimizer_ablation_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_optimizer_ablation_fixtures
```

## What Landed

`psionic-train` now owns one typed optimizer-ablation packet that binds:

- the frozen executor decision-threshold packet
- the frozen weekly review workflow packet
- the frozen unified throughput packet

The retained packet keeps one narrow claim explicit:

- the same-budget 4080 optimizer variant is promising enough to keep for the
  later `trained-v1` candidate path
- the lane reran it once because the first result cleared the frozen noise band
- the retained repeat still clears the frozen noise band
- the lane kept zero exactness, held-out, and adversarial regressions while it
  did that

## Current Retained Truth

- packet digest:
  `15ccd8383a89bb58e7aae003500710ba3d778abcd47a5c7840ccd8c54e6676a7`
- baseline model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- candidate model id:
  `tassadar-article-transformer-trace-bound-trained-v0-optimizer-ablation-candidate-v1`
- same-budget profile id:
  `local_4080_cuda_tailnet_x86_64`
- current-best row id:
  `psion_executor_local_cluster_ledger_row_4080_v1`
- decision-threshold digest:
  `fd669205c6874d107162ac82f760fcc4b2e95de803cb39a6fdb267dd5e751ec0`
- review-workflow digest:
  `c11b48bb9cb4381ccba810b5c154ffad6014c3b130c539a32b43dff4298078bf`
- unified-throughput digest:
  `ff12ece15c7917e2c430cb139d81c36c2d9e2964f9ee8197275664314fc037a7`
- baseline optimizer id:
  `adamw_beta2_0.95_weight_decay_0.10_eps_1e-08`
- candidate optimizer id:
  `adamw_beta2_0.98_weight_decay_0.10_eps_1e-08`
- initial promising run id:
  `tailrun-home-admitted-20260329a`
- repeat confirmation run id:
  `tailrun-home-admitted-20260329b`
- `reference_linear` delta above baseline:
  `73679.40346165793`
- `reference_linear` minimum meaningful delta:
  `65060.100461467104`
- `hull_cache` delta above baseline:
  `224815.26180754835`
- `hull_cache` minimum meaningful delta:
  `209005.1178118226`
- minimum `hull_cache` speedup improvement:
  `0.059403490993948926`
- minimum `hull_cache` speedup floor:
  `0.05`
- maximum CPU-gap reduction:
  `0.06276315967336421`
- maximum CPU-gap improvement floor:
  `0.05`
- baseline training steps per second:
  `82.40252049829174`
- candidate training steps per second:
  `87.611924143902`
- training steps-per-second delta:
  `5.209403645610266`
- exactness regression count:
  `0`
- held-out regression count:
  `0`
- adversarial regression count:
  `0`
- promotion posture:
  `retain_optimizer_for_trained_v1_candidate`

## Honest Current Meaning

This does not claim `trained-v1` is already promoted.

It does claim something narrower and useful:

- the first same-budget optimizer lever now has one retained winning packet
- the retained repeat stayed outside the frozen noise band
- the result is strong enough to keep as input into the later `trained-v1`
  promotion packet
- the lane still has to finish the rest of EPIC 8 before replacement claims are
  honest

## Validation

- `cargo run -q -p psionic-train --example psion_executor_optimizer_ablation_fixtures`
- `cargo test -q -p psionic-train psion_executor_optimizer_ablation -- --nocapture`
