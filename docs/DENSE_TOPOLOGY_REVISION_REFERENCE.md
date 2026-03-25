# Dense Topology Revision Reference

> Status: canonical `#534` dense topology-revision record, updated 2026-03-25
> after landing the runnable harness in
> `scripts/check-dense-topology-revision-contract.sh`.

This document records the first controlled topology-revision contract for the
cross-provider dense cluster.

## What Landed

The issue added `crates/psionic-train/src/dense_topology_revision_contract.rs`
with:

- one explicit distinction between hot replace-rank revisions and
  checkpoint-barrier grow or shrink revisions
- typed data-ordering policy per revision: replay continuation, checkpoint
  barrier reseed, or refusal
- typed checkpoint transition policy per revision: latest-checkpoint reuse,
  checkpoint-barrier reshard, or refusal
- operator and finalizer actions for supported and refused revisions
- direct bindings back to the whole-program run graph, dense recovery contract,
  and distributed checkpoint contract

## Canonical Runner

Run the harness from the repo root:

```bash
scripts/check-dense-topology-revision-contract.sh
```

## Current Scope

The current contract admits:

- hot replace-rank at the same world size
- grow-world through a checkpoint barrier and explicit reshard plan
- shrink-world through a checkpoint barrier and explicit reshard plan

The current contract still refuses:

- live remove-without-replacement
- hidden autoscaling
- generic live elastic membership

## Why The Split Matters

The current distributed data-feed substrate still only proves hot replace-rank.
Grow and shrink are only honest today if they stop at a durable checkpoint
barrier, reseed ordering for the new world size, and restore through an
explicit reshard plan. This contract freezes that difference instead of
blurring all three operations into one fake elasticity story.
