# Cross-Provider Launch Contract Reference

> Status: canonical `XTRAIN-3` / `#519` record, updated 2026-03-25 after
> landing the first provider-neutral launch-contract family in
> `crates/psionic-train/src/cross_provider_launch_contract.rs`.

This document records the shared runtime envelope above the current Google
single-node, Google swarm, RunPod, and local trusted-LAN launchers.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-cross-provider-launch-contracts.sh
```

## What Landed

`psionic-train` now owns one typed launch-contract family that freezes:

- runtime env
- artifact roots
- checkpoint roots
- metrics roots
- visualization roots
- finalizer expectations
- provider-specific projected launch, startup, and finalizer steps

The landed surface includes:

- `CrossProviderLaunchContract`
- `CrossProviderLaunchSourceBinding`
- `CrossProviderLaunchRuntimeEnvVar`
- `CrossProviderLaunchArtifactRoots`
- `CrossProviderLaunchStartupPlan`
- `CrossProviderLaunchFinalizerPlan`
- `CrossProviderProjectedStep`
- the binary `cross_provider_launch_contracts`
- the checker `scripts/check-cross-provider-launch-contracts.sh`
- the canonical fixtures under `fixtures/training/launch_contracts/`

## Canonical Lanes

The first launch-contract family keeps four existing lanes explicit:

- Google single-node accelerated `g2 + L4`
- Google configured-peer two-node swarm
- RunPod single-pod `8xH100`
- first local trusted-LAN swarm

Each contract binds one lane to:

- the root cross-provider program manifest id and digest
- one source binding
- one execution class
- one stable run id
- one shared set of runtime env vars
- one shared artifact-root layout
- one explicit startup plan
- one explicit finalizer plan
- one provider-specific projected step sequence

## Why This Matters

Before this issue, the repo had real launchers, startups, and finalizers, but
their training-facing semantics lived inside separate provider-specific shell
surfaces.

After this issue, the training system has one explicit runtime envelope above
those shell entrypoints. Provider-specific launchers remain the resource and
host adapters. The launch contract now owns the machine-legible training truth:

- what run id is being launched
- what execution class is being requested
- what env vars must exist
- what artifact roots must exist
- what cluster ports matter
- what startup surface must materialize
- what finalizer inputs and outputs are required

## Current Limits

This issue intentionally does not claim:

- provider API automation closure
- dense-rank runtime implementation
- app-surface changes
- mixed-backend dense launch closure

This issue closes the shared launch envelope first.
