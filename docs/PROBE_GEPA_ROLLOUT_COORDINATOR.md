# Probe GEPA Rollout Coordinator

Status: implemented_early

Psionic now owns an early coordinator contract for Probe GEPA rollout
optimization. This is distributed benchmark-driven optimization, not
distributed neural-network training. The coordinator evaluates text-bundle
candidates over benchmark tasks, records rollout evidence, updates the
candidate frontier, and retains a central reflection/proposal record.

The implementation lives in
`crates/psionic-train/src/probe_gepa_rollout_coordinator.rs`.

## Current Shape

The coordinator can run Stage 0 locally before any live Pylon dispatch. It
loads a Probe GEPA candidate manifest, builds rollout assignments, runs a local
deterministic evaluator backend, caches results, summarizes the candidate
frontier, and exports candidate refs for Probe, Omega, Artanis, and
benchmark-cloud.

The current Stage 0 target is 20 to 40 metric calls. The default local run uses
20 metric calls over retained and validation-shaped Terminal-Bench task refs.

## Pylon Boundary

Pylon remains a future evaluator backend for this issue. The coordinator
already models `pylon_pending` assignments, but it does not claim live Pylon
execution here. Live Pylon lease lifecycle integration belongs to the
OpenAgents/Omega assignment work.

The distributed shape is:

- Psionic coordinator selects a candidate and task set.
- Local evaluator runs first for deterministic proof.
- Later, Pylons receive independent metric-call rollout assignments.
- Rollout results return normalized score, artifact, proof, policy, resource,
  and failure-family refs.
- Psionic updates the candidate frontier and reflection/proposal records.

## Failure Semantics

Rollout status separates:

- `succeeded`
- `agent_failed`
- `infrastructure_failed`
- `policy_blocked`

This matters because GEPA should mutate candidate text for model/agent
failures, but infrastructure failures should not punish a candidate. Policy
violations stop advancement before rollouts run.

## Verification

Run:

```bash
cargo test -p psionic-train probe_gepa_rollout_coordinator --lib
```

The focused tests cover local Stage 0 metric calls, resumable cache behavior,
infrastructure versus agent failure classification, and policy-blocked
candidate refusal.
