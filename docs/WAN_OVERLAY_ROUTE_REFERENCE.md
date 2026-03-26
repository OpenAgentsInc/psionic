# WAN Overlay Route Reference

> Status: canonical `XTRAIN-29` / `#573` record, updated 2026-03-26 after
> landing the first internet-native route contract.

## What This Closes

Psionic now owns one explicit WAN route layer above the public registry and the
elastic device mesh.

The new contract lives in:

- `crates/psionic-train/src/wan_overlay_route_contract.rs`
- `crates/psionic-train/src/bin/wan_overlay_route_contract.rs`
- `fixtures/training/wan_overlay_route_contract_v1.json`
- `scripts/check-wan-overlay-route-contract.sh`

This issue closes the first truthful answer to:

- which peers are public-reachable versus NAT-constrained
- when Psionic chooses direct, relayed, or overlay transport
- which relay is actually admitted by the current mesh
- how one route fails over into another without operator folklore

## Contract Shape

The canonical contract freezes:

- one NAT posture record for every public registry record
- one retained route-quality sample for every canonical path
- one route record naming the selected transport, relay binding, and source
  selection basis
- one failover receipt proving a route transition for the same peer pair

## Current Canonical Internet Paths

The fixture keeps four routes explicit:

- `route.checkpoint_authority.google_runpod.direct`
  Google and RunPod stay on direct transport while both public endpoints remain
  healthy.
- `route.public_miner.local_rtx4080_local_mlx.relayed`
  the two local miner nodes start on a relay hop through Google because both
  peers are NAT-constrained.
- `route.public_miner.local_rtx4080_local_mlx.overlay_failover`
  the same local-node pair can fail over to an overlay tunnel when the relay
  path degrades.
- `route.checkpoint_authority.local_mlx_runpod.overlay`
  Apple MLX reaches RunPod through the overlay lane for later catch-up and
  checkpoint serving.

## First Failover Proof

The first route failover receipt is explicit:

- `failover.public_miner.local_rtx4080_local_mlx.1`

That receipt proves:

- the previous route was the relay-only contributor-window path
- the next route was the overlay-tunnel contributor-window path
- the trigger was packet-loss overflow

So Psionic can now say why the peer pair moved transports instead of leaving
the transition buried inside retry logs.

## Existing Psionic Binding

The route contract binds directly to:

- `DecentralizedNetworkContract`
- `PublicNetworkRegistryContract`
- `ElasticDeviceMeshContract`

That means every route remains grounded in:

- admitted node identity
- admitted public roles
- current active relay leases
- explicit membership revisions

## Pass Criteria

The contract is green only if all of the following stay true:

- every registry record keeps one NAT posture record
- the committed fixture matches generator output exactly
- direct, relayed, and overlay transports all remain present
- every relayed or overlay path names an active relay lease
- at least one failover receipt preserves a real transport transition for the
  same peer pair

## Current Limits

This issue intentionally does not claim:

- live checkpoint catch-up
- quantized outer sync
- public internet soak closure
- generic NAT punching beyond the admitted route set

This issue freezes route truth first: NAT posture, relay policy, overlay
selection, quality evidence, and typed failover receipts.
