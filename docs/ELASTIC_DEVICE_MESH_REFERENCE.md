# Elastic Device Mesh Reference

> Status: canonical `XTRAIN-28` / `#571` record, updated 2026-03-26 after
> landing the first elastic device mesh contract in
> `crates/psionic-train/src/elastic_device_mesh_contract.rs`.

This document records the first runtime-managed public mesh layer for the
decentralized Psionic training network.

## Canonical Runner

Run the contract checker from the repo root:

```bash
scripts/check-elastic-device-mesh-contract.sh
```

## What Landed

`psionic-train` now owns one mesh-runtime contract above the registry layer.

The landed surface includes:

- `ElasticDeviceMeshContract`
- `ElasticMeshRoleLeasePolicy`
- `ElasticMeshMemberLease`
- `ElasticMeshHeartbeatSample`
- `ElasticMeshDeathrattleNotice`
- `ElasticMeshRevisionReceipt`
- `write_elastic_device_mesh_contract(...)`
- the canonical fixture
  `fixtures/training/elastic_device_mesh_contract_v1.json`
- the checker
  `scripts/check-elastic-device-mesh-contract.sh`

## What The Contract Makes Explicit

The first mesh layer freezes these seams in one machine-legible object:

- one role-specific lease policy set
- one current lease set for miners, validators, checkpoint authorities, and the
  relay node
- one retained heartbeat sample set
- one explicit deathrattle notice instead of inferred disappearance
- one typed revision-receipt surface for activation, replacement, and refusal

## Current Canonical Mesh

The first canonical mesh keeps these current runtime facts explicit:

- Google, Apple MLX, and the departing RTX 4080 node retain public miner leases
- Google plus Apple MLX retain the active validator quorum
- Google plus RunPod retain active checkpoint-authority leases
- Google retains the single active relay lease

The first deathrattle-driven runtime replacement is also explicit:

- `local_rtx4080_workstation.registry` publishes a graceful departure notice
- `local_mlx_mac_workstation.registry` is the named replacement
- the applied revision
  `promote_public_miner_standby_after_deathrattle_v1` promotes the MLX node
  into the active public miner set

## Existing Psionic Binding

The mesh contract does not pretend to replace earlier topology or recovery
truth.

It binds directly to:

- `DecentralizedNetworkContract`
- `PublicNetworkRegistryContract`
- `DenseTopologyRevisionContract`
- `DenseRankRecoveryContract`

That means runtime-managed public-role replacement is explicit, but dense
world-size changes still stay bound to the older topology and recovery surfaces
until those contracts widen honestly.

## Honest Refusal Boundary

The first mesh layer also keeps one critical refusal explicit:

- `refuse_live_dense_world_change_without_checkpoint_barrier_v1`

That receipt binds directly to:

- `dense_topology.remove_rank_without_replacement.live_refused`
- `dense_rank.provider_loss.rank3.shrink_world_refused`

So the mesh now proves graceful public-role replacement while still refusing to
pretend full live dense elasticity already exists.

## Pass Criteria

The contract is green only if all of the following stay true:

- the committed fixture matches the generator output exactly
- role lease policies stay aligned with the current network cadence
- deathrattle and replacement stay explicit instead of hidden in operator
  notes
- the dense world-change refusal remains tied to the older refused topology and
  recovery cases

## Current Limits

This issue intentionally does not claim:

- full live dense world-size elasticity
- NAT traversal or overlay path selection
- WAN-grade communication policy
- public internet failure closure

This issue freezes elastic public-role mesh timing, heartbeats, deathrattles,
and revision truth first.
