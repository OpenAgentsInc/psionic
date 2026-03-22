# Psion Rented-Cluster Runbook

> Status: canonical `PSION-16` / `#372` rented-cluster contract, written
> 2026-03-22 after landing the pilot run and checkpoint-recovery bundle.

This document freezes the first repo-owned rented-cluster runbook for the
`Psion` learned-model lane.

It does not claim trusted-cluster scale-up is already complete. It does make
preemption, storage persistence, resume posture, cost guardrails, and explicit
infra refusal machine-legible before larger rented runs are honest.

## Canonical Artifacts

- `crates/psionic-train/src/psion_rented_cluster_runbook.rs` owns the rented-
  cluster runbook, infra-mode evaluations, stop-condition receipts, and the
  failure rehearsal bundle.
- `crates/psionic-train/examples/psion_rented_cluster_runbook_fixtures.rs`
  regenerates the canonical rented-cluster fixtures.
- `fixtures/psion/rented_cluster/psion_rented_cluster_runbook_v1.json` is the
  canonical rented-cluster runbook.
- `fixtures/psion/rented_cluster/psion_rented_cluster_failure_rehearsal_bundle_v1.json`
  is the canonical rented-cluster failure rehearsal bundle.

The stable schema versions are:

- `psion.rented_cluster_runbook.v1`
- `psion.rented_cluster_stop_condition_receipt.v1`
- `psion.rented_cluster_failure_rehearsal_bundle.v1`

## What The Runbook Freezes

The first rented-cluster runbook now binds:

- storage lifecycle profiles that keep checkpoints restorable on ephemeral
  hosts instead of treating local disks as durable truth
- scheduling and accounting policy for preemption handling and cost guardrails
- explicit support, downgrade, and refusal outcomes for named infra modes
- stop conditions that downgrade to resume-only posture on repeated preemption
  and stop the run on cost overrun
- a failure rehearsal bundle that cites the checkpoint-recovery restart and
  rollback receipts directly

## Mechanical Enforcement

`psionic-train` now validates that:

- checkpoint storage on rented clusters may not stay `ephemeral`; the runbook
  requires a restorable or immutable archive class
- log bundles stay explicitly ephemeral instead of pretending they carry the
  same durability contract as checkpoints
- downgraded infra modes must require
  `resume_from_last_stable_checkpoint`
- refused infra modes may not carry a fake recovery mode
- the rehearsal bundle proves preemption handling through typed admission
  receipts, checkpoint archive and cold-restore behavior through storage
  receipts, and cost guardrails through typed completion plus stop-condition
  receipts
- the rehearsal bundle must cite the forced-interruption restart and
  corruption-rollback receipts from `PSION-15`

## Claim Boundary

This issue still does **not** claim:

- trusted-cluster training is landed
- public-cluster or untrusted shared-cluster participation is allowed
- cross-region ephemeral clusters are supported
- best-effort retry loops can replace explicit rented-cluster stop or downgrade
  decisions

Later cluster issues must preserve this failure policy rather than widening the
infra posture silently.
