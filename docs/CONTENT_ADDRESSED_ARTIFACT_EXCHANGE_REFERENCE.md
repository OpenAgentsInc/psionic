# Content-Addressed Artifact Exchange Reference

> Status: canonical `XTRAIN-35` / `#579` record, updated 2026-03-26 after
> landing the first public artifact-exchange contract.

## What This Closes

Psionic now owns one content-addressed public artifact exchange surface above
the WAN, live catch-up, outer-sync, and remote-backend contracts.

The new contract lives in:

- `crates/psionic-train/src/content_addressed_artifact_exchange_contract.rs`
- `crates/psionic-train/src/bin/content_addressed_artifact_exchange_contract.rs`
- `fixtures/training/content_addressed_artifact_exchange_contract_v1.json`
- `scripts/check-content-addressed-artifact-exchange-contract.sh`

This issue closes the first truthful answer to:

- where public deltas, gradient slices, checkpoints, and provisional score
  artifacts live
- which artifacts are peer-seeded, relay-cached, or mirrored into
  authoritative stores
- how artifacts bind back to outer-sync receipts, live checkpoint
  advertisements, and validator assignment receipts
- how corruption stays explicit instead of being hand-waved away as a transient
  transport fault

## Contract Shape

The canonical contract freezes:

- five exchange backends
- five published artifacts
- five fetch receipts
- one refused digest-mismatch fetch

## Current Canonical Binding

The contract binds directly to:

- the public network registry contract
- the public work-assignment contract
- the WAN overlay route contract
- the live checkpoint catch-up contract
- the quantized outer-sync contract
- the remote train artifact backend contract set

That means content-addressed exchange now carries one machine-legible lineage
from source receipt or advertisement to fetch verification.

## Backend Classes

The first exchange layer explicitly separates:

- peer seeds for miner-owned deltas and gradient slices
- one Google relay cache for short-lived public artifact survival
- authoritative stores bound to the Google bucket backend and the RunPod
  workspace backend

This keeps public artifact movement from collapsing into one coordinator-owned
bucket or one machine-local directory walk.

## Artifact Families

The current published artifact set covers:

- two admitted quantized delta artifacts
- one retained raw gradient slice artifact
- one live checkpoint artifact
- one provisional validator score artifact

The score artifact is transport-truth only for now. Final validator verdict and
promotion semantics still belong to the later validator issues.

## Honest Corruption Boundary

The first explicit corruption refusal is:

- `fetch.gradient.google.runpod.refused`

That receipt proves a route can stay healthy while the artifact still fails
closed on digest mismatch.

## Pass Criteria

The contract is green only if all of the following stay true:

- artifact content ids recompute from kind, digest, and byte length
- delta and gradient artifacts still point at admitted outer-sync receipts
- the checkpoint artifact still points at the admitted live checkpoint
  advertisement
- the provisional score artifact still points at a validator assignment receipt
- exactly one refused digest-mismatch fetch remains explicit

## Current Limits

This issue intentionally does not claim:

- final validator verdict semantics
- checkpoint promotion or multi-validator consensus
- the full public miner execution protocol

This issue freezes transport truth first: artifact ids, backend classes, fetch
verification, and corruption refusal.
