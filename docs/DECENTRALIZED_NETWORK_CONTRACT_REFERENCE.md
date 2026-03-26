# Decentralized Network Contract Reference

> Status: canonical `XTRAIN-25` / `#569` record, updated 2026-03-26 after
> landing the first typed decentralized network contract in
> `crates/psionic-train/src/decentralized_network_contract.rs`.

This document records the first decentralized network contract in Psionic.

## Canonical Runner

Run the contract checker from the repo root:

```bash
scripts/check-decentralized-network-contract.sh
```

## What Landed

`psionic-train` now owns one typed public-network contract above the retained
cross-provider training-program manifest and whole-program run graph.

The landed surface includes:

- `DecentralizedNetworkContract`
- `DecentralizedNetworkRoleClass`
- `DecentralizedNetworkRoleBindingKind`
- `DecentralizedNetworkGovernanceRevision`
- `DecentralizedNetworkEpochCadence`
- `DecentralizedNetworkSettlementBackend`
- `DecentralizedNetworkCheckpointAuthorityPolicy`
- `write_decentralized_network_contract(...)`
- the canonical fixture
  `fixtures/training/decentralized_network_contract_v1.json`
- the checker
  `scripts/check-decentralized-network-contract.sh`

## What The Contract Makes Explicit

The first contract freezes these seams in one machine-legible object:

- one stable decentralized network id
- one explicit governance revision id and revision number
- one explicit registration posture for the first network
- one explicit fixed-window epoch cadence
- one explicit settlement backend posture
- one explicit checkpoint-authority policy with validator quorum
- one explicit public role set:
  `public_miner`, `public_validator`, `relay`, `checkpoint_authority`, and
  `aggregator`
- one explicit binding from those public roles back to the current
  program-manifest and run-graph vocabulary

## Existing Psionic Binding

The contract does not invent a second hidden root authority.

It binds directly to:

- `CrossProviderTrainingProgramManifest`
- `CrossProviderProgramRunGraph`
- `SharedValidatorPromotionContract`

The direct public-role bindings are:

- `public_miner` -> `validated_contributor_window`
- `public_validator` -> `validator`
- `checkpoint_authority` -> `checkpoint_writer`
- `aggregator` -> `data_builder`

`relay` remains explicit as a network-only support role in this issue. The
contract freezes that vocabulary without claiming the current run graph already
has a dedicated relay execution class.

## Pass Criteria

The contract is green only if all of the following stay true:

- the committed fixture matches the generator output exactly
- the contract stays bound to the canonical cross-provider program manifest
- the contract stays bound to the canonical whole-program run graph digest
- the contract stays bound to the shared validator and promotion contract id
- the public role set stays explicit and machine-legible
- relay remains an honest network-only support role until a dedicated execution
  class actually lands

## Current Limits

This issue intentionally does not claim:

- public node registration or wallet identity
- public network discovery or matchmaking
- live public miner or validator runtime
- public reward execution or payout settlement
- permissionless participation
- one dedicated relay or aggregator execution class in the current run graph

This issue freezes decentralized network epoch, role, governance, settlement,
and checkpoint-authority truth first.
