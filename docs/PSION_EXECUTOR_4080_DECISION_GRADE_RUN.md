# Psion Executor 4080 Decision-Grade Run

> Status: canonical `PSION-0306` / `#730` record, updated 2026-03-30 after
> landing the first retained 4080 decision-grade packet plus shared v2
> dashboard visibility packet.

This document records the first decision-grade 4080 packet that the admitted
Tailnet worker profile may cite for accelerator-backed executor questions.

## Canonical Fixtures

- `fixtures/psion/executor/psion_executor_4080_decision_grade_run_v1.json`
- `fixtures/training_visualization/psion_executor_4080_decision_grade_remote_training_visualization_bundle_v2.json`
- `fixtures/training_visualization/psion_executor_4080_decision_grade_remote_training_run_index_v2.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_4080_decision_grade_run_fixtures
```

## What Landed

`psionic-train` now owns one typed 4080 decision-grade packet that binds:

- the retained 4080 remote-launch packet
- the retained 4080 durable-checkpoint packet
- the retained 4080 frequent-eval attachment packet
- the retained 4080 smoke packet
- the retained 4080 interruption-recovery packet
- the retained admitted-device matrix report
- the retained same-node CUDA report and portable bundle
- one retained decision-grade run-registration row
- one retained weekly ablation review row
- one executor-specific remote-training visualization `v2` bundle
- one executor-specific remote-training run index `v2`

That means the first 4080 decision-grade run is no longer just a CUDA benchmark
report plus a separate Tailnet smoke packet. It is now one reviewable packet
that keeps baseline-comparable accelerator facts, roundtrip/recovery lineage,
frozen-pack binding, weekly review, and operator-surface visibility in the same
grammar.

## Current Retained Truth

- packet digest:
  `3e406cd85a6bf918fe0e219dc2e77f4801750798bc16aa28ca355457e44ae8f5`
- visualization bundle digest:
  `969bd18713eae8f2b7ee31aefc0ef842a4a6d7c9e94a4fb71b1a2715baa4cb41`
- run-index digest:
  `4a7b839e41e02d7cf0dc78a6b4de60f53e0049fd7ab1b0ce7109f01b95664167`
- approved equivalent-subset digest:
  `177adce779214af9a012fe647fff1e7867cf88d06677df0004759d8f2b44f2df`
- run-registration digest:
  `baf46f827951453770aee2f323d05bb456d9a233420e8bd3c40b2f29e88d366e`
- weekly review digest:
  `80b4747cb7002e292db5bd94a229905f0b850fd28b1f8bf8d5c1b0f49fade986`
- decision matrix run:
  `tailrun-admitted-device-matrix-20260327b`
- retained CUDA run:
  `same-node-wallclock-retained-cuda`
- supporting Tailnet run:
  `tailrun-home-admitted-20260328k`
- retained device:
  `cuda:0`
- completed steps:
  `47104`
- observed wallclock milliseconds:
  `571633`
- final mean loss:
  `0.0`
- retained dashboard entry count:
  `7`

## Approved Equivalent Checkpoint Subset

This packet does **not** pretend that the retained 4080 run already has two
fully repeated explicit checkpoint-pack snapshots inside the same admitted
cluster workflow.

Instead it uses one approved equivalent subset for the first accelerator-backed
decision-grade question:

- `frozen_pack_binding_green`
- `baseline_comparable_green`
- `local_cluster_roundtrip_green`
- `weekly_ablation_review_green`
- `dashboard_visibility_green`

That equivalent subset is allowed only because:

- the retained CUDA run consumes most of the admitted `600` second budget
- the explicit frequent-pack ledger row remains retained on the supporting
  Tailnet run
- the supporting Tailnet run keeps launch, checkpoint, replay, and recovery
  truth green
- the decision-grade packet now carries one explicit weekly ablation review row
- the run is visible in the shared remote-training `v2` bundle and run-index
  family instead of hiding behind isolated packet prose

## Claim Boundary

This packet counts only for the first admitted **4080 decision-grade** executor
question.

It does **not** claim:

- full promotion-pack scoring on a trained executor candidate
- multi-checkpoint repetition beyond the approved equivalent subset
- broader executor replacement readiness
- EPIC 4 canonical run-registration or searchable-ledger closure

Those remain later obligations.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_4080_decision_grade_run_fixtures`
- `cargo test -q -p psionic-train builtin_executor_4080_decision_grade_packet_is_valid -- --exact --nocapture`
- `cargo test -q -p psionic-train executor_4080_decision_grade_fixture_matches_committed_truth -- --exact --nocapture`
