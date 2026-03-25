# Cross-Provider Program Run Graph Reference

> Status: canonical `#532` whole-program cross-provider run-graph record,
> updated 2026-03-25 after landing the runnable harness in
> `scripts/check-cross-provider-program-run-graph.sh`.

This document records the first Psionic artifact that carries one pretraining
program across dense ranks, contributor windows, validators, checkpoint
writers, eval workers, and data builders under one shared run id.

## What Landed

The issue added `crates/psionic-train/src/cross_provider_program_run_graph.rs`
with:

- one typed whole-program run-graph artifact built on the existing
  `TrainingRunState` and `TrainingOrchestratorState`
- synthetic but explicit per-role participants for dense ranks, validated
  contributor windows, validators, checkpoint writers, eval workers, and data
  builders
- one whole-program role window that keeps active, standby, and quarantined
  roles machine-legible
- explicit evidence bindings back to retained provider-neutral execution
  segments or manifest-owned dataset authority
- a typed transition log so final evidence can reconstruct role composition
  without splitting the job into hidden provider-local program ids

## Canonical Runner

Run the harness from the repo root:

```bash
scripts/check-cross-provider-program-run-graph.sh
```

## Pass Criteria

The whole-program run graph is green only if all of the following are true:

- one run id carries every admitted execution class
- the orchestrator still owns its normal contributor-window planning
- dense, validator, checkpoint, eval, and data-builder roles remain explicit
  instead of being flattened into contributors
- retained evidence bindings can reconstruct the full role composition later

## Current Shape

The canonical artifact keeps these boundaries explicit:

- dense ranks remain active under the whole-program role window
- contributor windows are the only nodes selected into the current orchestrator
  rollout window
- one validator is active and one validator is quarantined under the shared
  validator contract
- one checkpoint writer is active and one remains standby for later recovery or
  topology revision
- data builders stay inside the same run id even though provider-neutral final
  evidence still reconstructs them from dataset authority rather than a
  dedicated execution segment

## Current Limitations

This issue intentionally does not claim:

- dense recovery after node or provider loss
- elastic world-size closure
- a retained multi-provider dense proof run
- dedicated provider-neutral data-builder execution segments

Those remain follow-on issues. This issue makes the whole-program participant
model real first.
