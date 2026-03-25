# Remote Train Artifact Backend Reference

> Status: canonical `XTRAIN-8` / `#524` record, updated 2026-03-25 after
> landing the first provider-neutral remote artifact backend contract in
> `crates/psionic-train/src/remote_artifact_backend_contract.rs`.

This document records the first shared remote artifact backend layer above the
older local retention and archival controller.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-remote-train-artifact-backend-contract.sh
```

## What Landed

`psionic-train` now owns:

- a provider-neutral `RemoteTrainArtifactBackend` trait
- one concrete Google bucket backend
- one concrete RunPod workspace-mirror backend
- typed placement decisions for checkpoints, logs, metrics, and final evidence
- typed finalizer projections that replace bespoke provider-root walking
- the generator binary `remote_train_artifact_backend_contract`
- the checker `scripts/check-remote-train-artifact-backend-contract.sh`
- the fixture `fixtures/training/remote_train_artifact_backend_contract_v1.json`

## Current Honest Boundary

This issue closes remote backend identity, placement policy, restore policy,
and finalizer projection for the current Google and RunPod train lanes.

It does not close:

- generic multi-cloud object-store clients
- mixed-backend checkpoint portability
- billing dashboards or operator spend UI
- local-workstation remote storage authority

What it proves now:

- Google and RunPod retain one shared typed artifact backend contract surface
- placement and restore policy are byte-accounted and machine-legible
- finalizers can read typed remote projections instead of bespoke provider
  directory layouts
