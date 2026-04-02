# Psion Actual Pretraining Systems Bundle

> Status: canonical actual-lane systems bundle, written 2026-04-02 after
> binding CS336 A2-style profiling, efficiency, distributed-runtime, and
> resume-support concerns into `psion_actual_pretraining_v1`.

This document freezes the machine-readable systems authority for the canonical
actual `Psion` pretraining lane.

It does not claim that later hardware admission gates, durable backup
automation, automatic eval triggers, or live dashboards are already done. It
does bind one A2-shaped systems surface directly into the actual lane so later
hardening issues consume a frozen baseline instead of ad hoc benchmarks.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_systems_bundle.rs` owns
  the typed systems-bundle contract.
- `crates/psionic-train/examples/psion_actual_pretraining_systems_bundle_fixtures.rs`
  regenerates the committed systems bundle.
- `fixtures/psion/pretrain/psion_actual_pretraining_systems_bundle_v1.json`
  carries the canonical actual-lane systems bundle.
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
  is the bounded trusted-cluster anchor the systems bundle cites.
- `docs/PSION_CHECKPOINT_RECOVERY.md`
  is the checkpoint and restart surface the systems bundle binds into resume
  support.

Stable schema version:

- `psion.actual_pretraining_systems_bundle.v1`

## What The Bundle Freezes

The systems bundle binds:

- one trusted-cluster throughput anchor from the broader-pretraining
  observability receipt
- one memory-headroom and checkpoint-size qualification for the admitted
  four-worker H100 shape
- one distributed-runtime qualification carrying backend, transport,
  collective benchmark digest, replay exactness, and distributed-step contract
  digest
- one hardware-preflight blocker set for backend family, worker inventory,
  storage credentials, and checkpoint restore
- one benchmark-binding set for throughput, collective sync, exact replay, and
  resume recovery
- one resume-support surface that names the accepted pointer path, the
  continuation handoff path, and the required restart/rollback/invalidation
  drills

## Why This Exists

The actual lane already had:

- one lane id
- one recipe and topology/storage bundle
- one evidence contract
- one launcher and resume contract
- one continuation handoff contract

What it did not have was one explicit place where systems work derived from
CS336 A2 could land without drifting into a detached curriculum lane.

This bundle closes that gap by turning systems profiling and distributed
qualification into a committed actual-lane contract. The launcher now loads it
as part of the frozen contract set, and later hardening work can consume it
directly.

## Current Claim Boundary

This bundle honestly claims:

- the actual lane has one retained throughput and memory baseline
- the actual lane has one admitted distributed-runtime qualification surface
- the actual lane has one explicit hardware-preflight blocker set
- the actual lane has one explicit resume-support baseline anchored to trusted
  checkpoint recovery drills

It does not yet claim:

- live launch refusal based on those preflight items
- automatic backup or auto-resume closure
- automatic checkpoint eval triggering
- dashboards or alert routing
- a final end-to-end actual-lane rehearsal
