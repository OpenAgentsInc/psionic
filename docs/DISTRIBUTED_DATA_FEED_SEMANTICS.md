# Distributed Data-Feed Semantics

> Status: canonical `PLIB-219` / `#3734` reference record, updated 2026-03-16
> after landing the first bounded distributed data-feed report in
> `crates/psionic-data/src/lib.rs`.

This document records the current bounded distributed and sharded data-feed
surface for Psionic.

## Canonical Runner

Run the distributed data-feed harness from the repo root:

```bash
scripts/release/check-psionic-distributed-data-feed-semantics.sh
```

## What Landed

`psionic-data` now exposes:

- `DistributedSamplerPartitionContract`
- `DistributedWorkerCoordinationContract`
- `DistributedReplayOrderingContract`
- `DistributedDataFeedContract`
- `DistributedDataFeedPlan`
- `DistributedDataFeedSemanticsReport`
- `builtin_distributed_data_feed_semantics_report()`
- `TopologyRevisableDistributedWorkerMember`
- `TopologyRevisableDataFeedRevisionRequest`
- `TopologyRevisableShardOwnershipReassignment`
- `TopologyRevisableDistributedDataFeedReceipt`
- `TopologyRevisableDistributedDataFeedSemanticsReport`
- `builtin_topology_revisable_distributed_data_feed_semantics_report()`

## Current Honest Posture

Today Psionic has a first-class bounded distributed data-feed surface, but it
does **not** claim elastic or fault-tolerant distributed trainer closure yet.

The bounded seeded surface now makes these seams explicit:

- fixed-world-size shard partitioning with contiguous-block and rank-strided
  assignment
- epoch-barrier and fixed-cadence step-barrier worker coordination
- runtime-derived replay-safe per-rank ordering through
  `RuntimeDeterminismContract`
- explicit refusal for elastic membership or rebalance-aware partitioning
- explicit support for same-world-size rank replacement with typed shard
  ownership reassignment and replay-continuity receipts
- explicit refusal for grow-world, shrink-world, or remove-without-replacement
  revisions

## Why This Matters

This report prevents two failure modes:

- implying that local data ingress automatically means distributed feed truth
- implying that fixed-world-size shard coordination already solves elastic
  membership, topology revision, or full distributed run control

The point of this issue is to make bounded distributed data-feed behavior a
reusable library contract that later distributed runtime work can extend
honestly.

## Topology-Revisable Dense Scope

The current cross-provider extension is still narrow by design.

What it supports now:

- replacing one departed dense rank with one new node at the same rank
- preserving the same replay-safe global shard order before and after the
  replacement
- emitting typed shard ownership reassignment records from the departed node to
  the replacement node
- keeping the world size fixed while the replacement is admitted

What it still refuses:

- grow-world revisions
- shrink-world revisions
- removal without replacement
- hidden elastic membership behind the fixed-world-size contract

Run the focused checker from the repo root:

```bash
scripts/check-topology-revisable-distributed-data-feed.sh
```
