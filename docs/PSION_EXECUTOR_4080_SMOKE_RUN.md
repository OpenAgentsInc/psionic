# Psion Executor 4080 Smoke Run

> Status: canonical `PSION-0304` / `#728` record, updated 2026-03-30 after
> landing the first retained accelerator-backed smoke-run packet for the
> admitted Mac -> 4080 Tailnet executor lane.

This document records the first retained 4080 smoke-run packet that binds one
real accelerator-backed training run to explicit checkpoint, throughput,
memory, eval, and failure facts instead of relying on launch-only or
checkpoint-only evidence.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_4080_smoke_run_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_4080_smoke_run_fixtures
```

## What Landed

`psionic-train` now owns one typed smoke-run packet that binds:

- the prerequisite frequent-eval attachment packet
- the retained admitted run summary
- the retained coordinator and contributor runtime reports
- the Linux 4080 contribution row
- the dataset slices used by the retained smoke objective
- the checkpoint pointer and frequent-pack ledger row for the same run
- the explicit failure and promotion-block facts that kept the run honest

That means the admitted 4080 lane now has one explicit packet for:

- first real accelerator-backed smoke-run evidence
- one retained checkpoint tied to the Linux 4080 worker contribution
- one retained throughput and memory fact set
- one retained frequent-pack attachment row
- one retained failure record explaining why this run did not count as a
  promotable executor candidate

## Current Retained Truth

- packet digest:
  `2f13de7c485046bb3f478fcfff4711f4e0b033e01d1df1ff3950af55f61b128c`
- frequent-eval attachment packet SHA256:
  `93db62e514bdb276426fcddea240543e1d42da077dee4756777dd0bea7f89e83`
- retained run summary SHA256:
  `fe6e0fb07458810923b6f96a7830636216e741ba49ae612d9ba54847696a7dd4`
- retained run summary digest:
  `e37cd022b7082183a5a75014dca3d2512abfa18a5ce4355140c8b4cd9c5ac73b`
- coordinator report SHA256:
  `dca06156d1c590d4343959f375d1572879540fe29071f0426c93eec7171b9c5c`
- contributor report SHA256:
  `f1ca4595a4b3c77f037b519d62048508a426c38ccf4393054218d1d4039c84bc`
- run id:
  `tailrun-home-admitted-20260328k`
- run family id:
  `swarm.local.mlx_metal_plus_rtx4080.open_adapter.v1`
- smoke objective id:
  `psion.executor.4080_tailnet_smoke_objective.v1`
- smoke objective kind:
  `executor_lane_local_cluster_admission_surrogate`
- objective dataset id:
  `dataset://openagents/swarm/open_adapter_sft@2026.03.24`
- objective split:
  `train`
- objective slice ids:
  `swarm_mac_mlx_coordinator_validator_contributor-1`,
  `swarm_linux_cuda_rtx4080_contributor-2`
- result classification:
  `bounded_success`
- Linux worker id:
  `swarm-linux-4080-a`
- backend:
  `open_adapter_backend.cuda.gpt_oss_lm_head`
- worker endpoint:
  `100.108.56.85:34101`
- observed wallclock:
  `2945 ms`
- local execution wallclock:
  `3 ms`
- executed steps:
  `12`
- batch count:
  `2`
- sample count:
  `12`
- payload bytes:
  `1480`
- final mean loss:
  `5.364421e-07`
- estimated steps per second:
  `4000.0`
- estimated samples per second:
  `4000.0`
- contribution share:
  `0.5`
- free memory bytes at admission:
  `21474836480`
- accelerator count at admission:
  `1`
- contributor receipt digest:
  `865ef2f86a2b50a6790996a07f53a345068eea2514cc7907c47bdece2c4c6305`
- checkpoint family:
  `swarm.local.open_adapter.policy:tailrun-home-admitted-20260328k`
- checkpoint pointer digest:
  `dd1aa85c355eb43934a20afe7e4204b3ed82bb85f4fe392dfde45229f4e434f8`
- checkpoint ref:
  `checkpoint://swarm/first-swarm-live-plan/policy`
- eval ledger row id:
  `psion.executor.4080.frequent_eval_row:tailrun-home-admitted-20260328k:dd1aa85c355eb43934a20afe7e4204b3ed82bb85f4fe392dfde45229f4e434f8`
- eval ledger row digest:
  `7b9b1159894d59c216bf4e8c069036a9c2820b9f949a7298d13ba93d4a52eaa4`
- operator-review suite status:
  `green`
- missing eval blocks promotion:
  `true`
- publish disposition:
  `refused`
- promotion disposition:
  `held`
- unsupported precision refusal:
  `open adapter backend does not yet support precision policy \`Bf16Mixed\``
- accepted contributions:
  `2`
- replay-checked contributions:
  `2`
- submission receipt count:
  `2`
- merge strategy:
  `exact_mean_delta_rank_stacking`
- merged LoRA rank:
  `4`
- canonical profile mean loss:
  `1.788139627478813e-7`
- deterministic probe top token id:
  `2`

## Claim Boundary

This packet counts as the first admitted **4080 smoke-run packet** for the
executor lane.

It does **not** claim:

- decision-grade readiness on the 4080 lane
- green exactness or held-out executor scoring on this open-adapter rerun
- that the unsupported precision refusal has already been resolved
- that a publish or promotion path is open for this retained smoke run

Instead it proves that one real accelerator-backed retained run exists, writes
one retained checkpoint, keeps one explicit frequent-pack row, records
throughput and memory facts, and keeps the blocker facts visible.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_4080_smoke_run_fixtures`
- `cargo test -q -p psionic-train psion_executor_4080_smoke_run -- --nocapture`
