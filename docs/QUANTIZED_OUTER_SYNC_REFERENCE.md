# Quantized Outer Sync Reference

> Status: canonical `XTRAIN-31` / `#575` record, updated 2026-03-26 after
> landing the first WAN-aware outer-sync contract.

## What This Closes

Psionic now owns one explicit outer-sync layer above the live catch-up and WAN
route contracts.

The new contract lives in:

- `crates/psionic-train/src/quantized_outer_sync_contract.rs`
- `crates/psionic-train/src/bin/quantized_outer_sync_contract.rs`
- `fixtures/training/quantized_outer_sync_contract_v1.json`
- `scripts/check-quantized-outer-sync-contract.sh`

This issue closes the first explicit answer to:

- which quantized delta families are admitted over WAN links
- which exchanges were actually applied
- which checkpoint authority aggregated them
- which full-precision WAN path remains refused

## Contract Shape

The canonical contract freezes:

- three explicit delta policies
- three exchange receipts
- one aggregation receipt
- two correctness receipts

## Admitted WAN Delta Families

The fixture keeps three policy classes explicit:

- `policy.int8_pseudogradient_outer_sync`
- `policy.nf4_residual_outer_sync`
- `policy.fp16_dense_allreduce_refused`

That keeps both the admitted quantized paths and the refused dense all-reduce
path visible in one place.

## Applied Exchanges

The first applied WAN exchanges are:

- `exchange.public_miner.google_to_runpod.int8.1`
- `exchange.public_miner.local_mlx_to_runpod.nf4.1`

Both are anchored to the completed MLX catch-up receipt, so the contract says
explicitly that public WAN outer sync begins only after the replacement miner
has rejoined truthfully.

## Honest Refusal

The first refused WAN path is:

- `exchange.public_miner.local_mlx_to_runpod.fp16_refused.1`

That receipt preserves one important boundary:

- full-precision dense all-reduce over the WAN overlay is not admitted

So later reports cannot quietly collapse quantized outer sync back into a
provider-LAN-only communication story.

## Existing Psionic Binding

The outer-sync contract binds directly to:

- `ElasticDeviceMeshContract`
- `WanOverlayRouteContract`
- `LiveCheckpointCatchupContract`

That means delta exchange stays tied to:

- admitted public-miner membership
- admitted checkpoint-authority destinations
- admitted route ids
- admitted live catch-up anchors

## Pass Criteria

The contract is green only if all of the following stay true:

- at least one applied quantized WAN exchange remains present
- at least one refused full-precision WAN exchange remains present
- aggregation bandwidth equals the sum of applied compressed bytes
- correctness receipts remain attached to applied quantized exchanges

## Current Limits

This issue intentionally does not claim:

- validator-graded incentive settlement
- public internet fault-injection closure
- soak-tested decentralized-runtime promotion

This issue freezes the first WAN-scaled synchronization truth first: delta
policy, applied exchange, refused exchange, aggregation, and correctness.
