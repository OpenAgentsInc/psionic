# Internet Fault Harness Reference

> Status: canonical `XTRAIN-32` / `#576` record, updated 2026-03-26 after
> landing the first internet fault and soak harness contract.

## What This Closes

Psionic now owns one explicit realism harness above the route, catch-up, and
outer-sync contracts.

The new contract lives in:

- `crates/psionic-train/src/internet_fault_harness_contract.rs`
- `crates/psionic-train/src/bin/internet_fault_harness_contract.rs`
- `fixtures/training/internet_fault_harness_contract_v1.json`
- `scripts/check-internet-fault-harness-contract.sh`

This issue closes the first explicit answer to:

- which public-internet faults are injected repeatedly
- which throughput baselines matter for catch-up and outer sync
- which suites must pass before promotion claims are allowed
- which case still forces a truthful hold

## Contract Shape

The canonical contract freezes:

- four fault profiles
- three throughput baselines
- two soak suites
- seven retained run receipts

## Fault Profiles

The fixture keeps these profiles explicit:

- `profile.packet_loss_public_miner_failover`
- `profile.delayed_checkpoint_catchup`
- `profile.bandwidth_throttle_outer_sync`
- `profile.validator_loss_hold`

That means public-runtime claims are no longer allowed to hide behind one
generic “bad network” bucket.

## Retained Passes

The harness now retains repeated passed evidence for:

- route failover under packet loss
- delayed catch-up under stretched checkpoint delivery
- throttled quantized outer sync with aggregation closure

Those passes appear in both the day matrix and the night soak suite.

## Honest Hold

The harness also retains one held result:

- `run.validator_loss_hold.day1`

That receipt keeps one important truth explicit:

- removing the Apple MLX validator breaks the current two-validator quorum
- the run holds instead of quietly pretending quorum survived

## Existing Psionic Binding

The fault harness binds directly to:

- `ElasticDeviceMeshContract`
- `WanOverlayRouteContract`
- `LiveCheckpointCatchupContract`
- `QuantizedOuterSyncContract`

That means fault and soak claims stay tied to:

- admitted route failovers
- admitted catch-up completions
- admitted outer-sync exchanges and aggregation
- admitted validator quorum shape

## Pass Criteria

The contract is green only if all of the following stay true:

- the day fault matrix keeps at least three passed runs
- the night soak suite keeps at least three passed runs
- the packet-loss case retains a real failover receipt
- the catch-up case retains a completed catch-up receipt
- the bandwidth-throttle case retains applied outer-sync plus aggregation
- the validator-loss case remains held

## Current Limits

This issue intentionally does not claim:

- incentive settlement
- miner reward accounting
- validator economics

This issue freezes the realism gate first: fault profiles, baselines, suite
thresholds, repeated passes, and one explicit hold condition.
