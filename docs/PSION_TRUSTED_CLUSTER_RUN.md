# Psion Trusted-Cluster Run

> Status: canonical `PSION-17` / `#373` trusted-cluster run contract, written
> 2026-03-22 after landing the first bounded multi-host Psion run bundle.

This document freezes the first truthful trusted-cluster training lane for
`Psion`.

It does not claim arbitrary public all-reduce, mixed-backend meshes,
cross-region fabrics, or elastic world-size growth are now closed. It records
one bounded four-worker trusted-cluster path with explicit topology,
throughput, replay, and checkpoint-restart receipts.

## Canonical Artifacts

- `crates/psionic-train/src/psion_trusted_cluster_run.rs` owns the
  trusted-cluster topology contract, distributed-group receipt, replay receipt,
  and full trusted-cluster run bundle.
- `crates/psionic-train/examples/psion_trusted_cluster_run_fixtures.rs`
  regenerates the canonical trusted-cluster fixtures.
- `fixtures/psion/trusted_cluster/psion_trusted_cluster_topology_contract_v1.json`
  is the canonical bounded topology contract.
- `fixtures/psion/trusted_cluster/psion_trusted_cluster_replay_receipt_v1.json`
  is the canonical multi-host replay receipt.
- `fixtures/psion/trusted_cluster/psion_trusted_cluster_checkpoint_recovery_bundle_v1.json`
  is the canonical trusted-cluster checkpoint-recovery bundle.
- `fixtures/psion/trusted_cluster/psion_trusted_cluster_run_bundle_v1.json`
  is the canonical self-contained trusted-cluster run bundle.

The stable schema versions are:

- `psion.trusted_cluster_topology_contract.v1`
- `psion.trusted_cluster_replay_receipt.v1`
- `psion.trusted_cluster_run_bundle.v1`
- `psion.checkpoint_recovery_bundle.v1` for the bound restart coverage

## What The Bundle Freezes

The first trusted-cluster bundle now binds:

- one broader pretrain-stage receipt and one broader-run observability receipt
- one bounded trusted-cluster topology contract for a homogeneous four-node
  CUDA H100 tensor-parallel lane with one rank per node
- one train-owned distributed-group receipt that freezes the public
  distributed-group truth this lane requires
- one distributed-optimizer contract and one realized distributed step receipt
- one exact multi-host replay receipt covering all four worker identities
- one checkpoint-recovery bundle that includes distributed restart coverage for
  the supported cluster path
- explicit refused or out-of-scope posture for mixed, cross-region, shared,
  and elastic-world-size modes

That makes trusted-cluster scale-up a repo-owned evidence bundle instead of a
cluster-shaped narrative claim.

## Mechanical Enforcement

`psionic-train` now validates that:

- the broader observability receipt still matches the bound broader-stage
  receipt
- the topology contract still points at the exact broader-stage,
  observability, and rented-cluster runbook digests it claims to preserve
- the realized cluster execution, training collective, and distributed-group
  receipt still agree on backend, transport, worker count, mesh identity, and
  member set
- the distributed-optimizer contract and realized step still carry the same
  collective plan and contract digest
- the replay receipt still covers all worker identities and still verifies as
  an exact match
- the checkpoint-recovery bundle still validates on the broader trusted-cluster
  run inputs and still includes a sharded distributed restart on the supported
  worker count

## Distributed Seam

The training-facing distributed-group receipt lives in `psionic-train` instead
of importing `psionic-distributed` directly.

That boundary is deliberate: `psionic-distributed` already depends on
`psionic-train`, so reversing the dependency would create a workspace cycle.
The receipt keeps the required public distributed-group truth explicit and
machine-checkable without pretending the cycle does not exist.
