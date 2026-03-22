# Psion Checkpoint Recovery

> Status: canonical `PSION-15` / `#371` checkpoint-recovery contract, written
> 2026-03-22 after landing the first pilot-stage, observability, and pilot-run
> receipts.

This document freezes the first bounded checkpoint-recovery contract for the
`Psion` learned-model lane.

It does not claim trusted-cluster full-model training is already complete. It
does make the required restart semantics explicit before rented-cluster or
trusted-cluster scale-up is honest.

## Canonical Artifacts

- `crates/psionic-train/src/psion_checkpoint_recovery.rs` owns the dense and
  sharded checkpoint artifact receipts, corruption receipts, recovery events,
  and full recovery bundle.
- `crates/psionic-train/examples/psion_checkpoint_recovery_fixtures.rs`
  regenerates the canonical recovery fixtures.
- `fixtures/psion/checkpoint_recovery/psion_dense_checkpoint_artifact_v1.json`
  is the canonical dense checkpoint artifact receipt.
- `fixtures/psion/checkpoint_recovery/psion_sharded_checkpoint_artifact_v1.json`
  is the canonical sharded checkpoint artifact receipt.
- `fixtures/psion/checkpoint_recovery/psion_checkpoint_recovery_bundle_v1.json`
  is the canonical recovery bundle across dense restart, sharded distributed
  restart, corruption rollback, and corruption invalidation.

The stable schema versions are:

- `psion.checkpoint_artifact.v1`
- `psion.checkpoint_corruption_receipt.v1`
- `psion.checkpoint_recovery_event.v1`
- `psion.checkpoint_recovery_bundle.v1`

## What The Bundle Freezes

The first recovery bundle now binds:

- one dense checkpoint artifact with pointer-first restart semantics
- one sharded checkpoint artifact for distributed restart of the same logical
  promoted checkpoint
- explicit optimizer-state restart contracts for both layouts
- one forced-interruption restart receipt
- one distributed restart receipt
- one corruption-triggered rollback receipt
- one corruption-triggered invalidation receipt

That keeps restart, rollback, and invalidation machine-legible instead of
leaving them as operator folklore.

## Mechanical Enforcement

`psionic-train` now validates that:

- the dense and sharded artifacts both bind to the same source promoted
  checkpoint label while keeping their own manifest and pointer truth explicit
- dense artifacts stay single-shard and sharded artifacts stay multi-shard
- optimizer-state restart receipts preserve step identity, parameter-group
  count, and exact sampling-cursor requirements
- artifact context carries dataset identity, sampling-policy identity, source
  checkpoint topology, and realized training hardware topology
- forced-interruption and distributed restarts recover through explicit restore
  receipts instead of hidden local heuristics
- corruption rollback must select the last stable artifact through listing
  fallback rather than silently continuing from corrupted state
- corruption invalidation is a first-class terminal outcome; corruption cannot
  remain a soft warning

## Claim Boundary

This issue still does **not** claim:

- trusted-cluster training is already landed
- arbitrary cluster-topology recovery is supported
- corrupted runs can be hand-fixed outside the recorded receipts
- broader pretraining beyond the bounded pilot-derived checkpoint contract

Later cluster issues must reuse this receipt surface rather than inventing a
looser recovery story.
