# Environment Runtime Service Reference

> Status: canonical bounded live-runtime record, updated 2026-04-01 after
> landing the first reusable environment runtime service above the existing
> package ABI and session machine.

This document records the first live environment runtime service in
`psionic-environments`.

## What Landed

The environment crate already owned:

- versioned package identity
- package validation
- deterministic runtime sessions
- registry install, pin, and composition logic

This issue adds the missing live execution layer:

- a reusable `EnvironmentRuntimeService`
- bounded worker-pool and queue policy
- typed submission, activation, and completion receipts
- queued, active, and completed execution state
- live execution over the existing `EnvironmentRuntimeSession`
- explicit workload-class admission for RL and eval paths
- dataset/sample cursor tracking through `EnvironmentTaskCursor`

## Canonical Shape

The service is intentionally bounded and library-first.

It owns:

1. package install through the existing registry
2. execution submission against one installed package version
3. typed refusal when the package is unknown, retired, unsupported, or the
   queue is full
4. worker activation when a queue item and worker are both available
5. execution of one declarative turn plan through the existing session state
   machine
6. completion receipts plus the final session summary

The current service makes these environment-runtime behaviors explicit:

- package loading
- session creation and teardown
- task iteration metadata
- prompt/turn input handling
- tool-call execution
- rubric-scored finalization
- RL-mode and eval-mode admission
- bounded concurrency and backpressure

## Canonical Runner

Run the focused harness from the repo root:

```bash
scripts/release/check-psionic-environment-runtime-service.sh
```

## Pass Criteria

The live runtime service is green only if all of the following remain true:

- installed package versions are resolved through the registry rather than
  free-form runtime strings
- unsupported workloads refuse before execution starts
- queue overflow refuses explicitly
- worker assignment is visible in activation receipts
- turn plans execute through the existing session state machine rather than a
  parallel ad hoc runner
- completion receipts preserve turn receipts and final session summary

## Current Boundary

This runtime service is still bounded.

It does not yet claim:

- remote multi-process worker execution
- sandbox-pool integration beyond the package/runtime contract boundary
- automatic task sampling from external data stores
- production durability or network RPC surfaces

It does close the missing in-process runtime-service layer above the environment
ABI, which was the real immediate gap.
