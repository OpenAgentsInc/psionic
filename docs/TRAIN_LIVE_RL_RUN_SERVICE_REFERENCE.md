# Training Live RL Run Service Reference

> Status: canonical bounded live-run-service record, updated 2026-04-01 after
> landing the first durable live RL run service in `psionic-train`.

This document records the first persistent live RL run service above the
existing train substrate.

## What Landed

`psionic-train` already owned:

- `TrainingRunState`
- `TrainingOrchestratorState`
- `RolloutWorkerProtocolState`
- `RolloutValidatorState`
- bounded `TrainingSamplerService`
- bounded `OpenAdapterLiveRlUpdateExecutor`

This issue adds the missing durable service layer above that substrate:

- a reusable `LiveRlRunService`
- durable per-run `state.json` persistence
- persistent `status.json` and per-window status artifacts
- failure artifacts under `failures/`
- run creation and graceful stop semantics
- window planning, activation, sealing, scoring, and reconciliation through the
  service
- worker heartbeat, claim, and rollout-upload flow against service-owned state
- validator bundle ingestion against service-owned validator state
- restart recovery by reloading run state from the durable service root

## Canonical Shape

The first service is intentionally bounded and library-first.

It owns:

1. run creation from cluster snapshot, target policy revision, and
   policy-weight broadcast
2. persistent run-graph plus orchestrator state under one service-owned root
3. one worker-protocol state machine per planned window
4. run status inspection and per-window progress inspection
5. rollout-worker heartbeats, claims, uploads, and validator verdict ingestion
6. trainer-batch assembly from live run state
7. graceful draining and terminal stop finalization

## Durable Layout

Each run is persisted under:

- `runs/<run_id>/state.json`
- `runs/<run_id>/status.json`
- `runs/<run_id>/windows/<window_id>.json`
- `runs/<run_id>/artifacts/*.json`
- `runs/<run_id>/failures/*.json`

This keeps the service honest about where live state actually lives. Restart
recovery loads `state.json` back into typed service state rather than
reconstructing from log lines.

## Current Boundary

This first live run service is still bounded.

It does not yet claim:

- network RPC or HTTP control-plane surfaces
- remote multi-process execution
- automatic environment sampling or sampler-driven rollout generation
- automatic trainer-step promotion inside the service loop

The current service closes the durability and lifecycle gap above the existing
run graph, orchestrator, worker protocol, and validator substrate. The
promotion path remains explicit through the separate live update bridge.

## Canonical Runner

Run the focused harness from the repo root:

```bash
scripts/release/check-psionic-train-live-rl-run-service.sh
```

That harness proves:

- bounded run creation
- window plan and activation
- worker heartbeat, claim, and upload flow
- validator bundle ingestion
- trainer-batch assembly
- graceful drain to completed stop
- restart recovery from the durable service root
- failure artifact emission on terminal stop
