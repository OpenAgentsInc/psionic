# Cross-Provider Runtime Binder Reference

> Status: canonical `XTRAIN-10` / `#526` record, updated 2026-03-25 after
> landing the provider-neutral runtime binder in
> `crates/psionic-train/src/cross_provider_runtime_binder.rs`.

This document records the shared binder that sits between the root
cross-provider training-program manifest and the concrete Google, RunPod, and
local training adapters.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-cross-provider-runtime-binder.sh
```

## What Landed

`psionic-train` now owns one typed runtime binder contract that freezes:

- the mapping from admitted compute sources to concrete launch contracts
- the shared runtime-env surface that each provider adapter must materialize
- the shared artifact backend projection each provider adapter must honor
- the provider-specific preflight, launch, startup, runtime, and finalizer
  hooks that remain in the adapter layer
- the stable runbook path that explains each bound lane

The landed surface includes:

- `CrossProviderRuntimeBinderContract`
- `CrossProviderRuntimeBindingRecord`
- `CrossProviderRuntimeHook`
- `CrossProviderBoundRuntimeEnv`
- `CrossProviderBoundArtifactClass`
- the binary `cross_provider_runtime_binder`
- the checker `scripts/check-cross-provider-runtime-binder.sh`
- the committed fixture `fixtures/training/cross_provider_runtime_binder_v1.json`

## Why This Matters

The launch-contract family already froze the shared runtime envelope. This
issue closes the next layer above it.

The runtime binder is the training-facing answer to one simple question:

- given one program manifest, one admitted compute source, and one admitted
  execution class, what exactly does the provider adapter have to materialize?

After this issue, the answer is machine-legible. Google, RunPod, and local
lanes keep their own resource hooks, but they do not own training truth.

The binder now owns:

- which launch contract is active
- which source contract is active
- which env vars are mandatory
- which artifact classes and remote backends are authoritative
- which startup and finalizer entrypoints the lane consumes
- which provider hooks still remain adapter-specific

## Current Limits

This issue intentionally does not claim:

- dense runtime closure
- provider API automation beyond the retained adapter hooks
- same-job mixed-backend dense training
- public-swarm discovery or adversarial internet participation

This issue closes the shared binder layer first.

