# Sharded Distributed Checkpoint Reference

> Status: canonical `XTRAIN-7` / `#523` record, updated 2026-03-25 after
> landing the first provider-neutral distributed checkpoint contract in
> `crates/psionic-train/src/distributed_checkpoint_contract.rs`.

This document records the first generic distributed checkpoint contract above
the older single-manifest pointer-first recovery layer.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-sharded-distributed-checkpoint-contract.sh
```

## What Landed

`psionic-train` now owns one typed distributed checkpoint surface with:

- root `CheckpointManifest` and `CheckpointPointer` authority objects
- explicit `DistributedCheckpointShardPlacement` records for parameter and
  optimizer shards
- explicit `DistributedCheckpointShardUploadReceipt` records covering durable
  uploads and refused partial uploads
- one deterministic `DistributedCheckpointRestorePlan` with dense-rank restore
  assignments
- the generator binary `sharded_distributed_checkpoint_contract`
- the checker `scripts/check-sharded-distributed-checkpoint-contract.sh`
- the fixture `fixtures/training/sharded_distributed_checkpoint_contract_v1.json`

## Current Honest Boundary

This issue closes sharded checkpoint authority and restore planning. It does
not close:

- same-job mixed-backend dense restore
- generic multi-store replication
- validator adjudication over dense checkpoint promotion
- cross-backend optimizer numeric parity

What it proves now:

- one provider-neutral checkpoint family can retain parameter and optimizer
  shards together
- partial shard uploads are refused explicitly instead of being treated as
  durable
- dense-rank restore assignments stay deterministic and machine-legible
- provider-specific storage and launcher details are not baked into restore
  logic
