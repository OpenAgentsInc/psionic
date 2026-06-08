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

The coordinator can also import live Omega/Pylon closeouts through
`ProbeGepaLiveCloseoutImport`. A live import enters the same completed-rollout
set, evaluation cache, frontier summary, and export surface used by the local
backend. The importer validates assignment id, campaign id, split ref, task id,
candidate id/hash, Probe commit, selected signatures, tool menu, verifier refs,
artifact/proof/resource refs, route scorecard ref, payment mode, closeout
state, public-claim posture, runtime-promotion claim, and model-training
authority claim before mutating state.

The current canonical live import fixture is the Probe GEPA Terminal-Bench 2
Pylon canary:

- assignment:
  `assignment.public.probe_gepa.terminal_bench_2.canary.20260608151057`
- Pylon: `pylon.artanis.gepa_stats_canary.20260608150415`
- Probe run:
  `probe_run.public.probe_gepa.terminal_bench_2.canary.20260608151057`
- closeout:
  `probe_closeout.probe_run.public.probe_gepa.terminal_bench_2.canary.20260608151057`

The Pylon worker closeout was accepted as evidence work. The benchmark import
enters Psionic as an `agent_failed` retained `service_readiness` rollout with
score `0`, no public benchmark score, no paid-work claim, no settlement claim,
no runtime promotion, and no model-training authority.

The current Stage 0 target is 20 to 40 metric calls. The default local run uses
20 metric calls over retained and validation-shaped Terminal-Bench task refs.

## Live Import Boundary

Pylon execution remains outside Psionic. Omega/OpenAgents own assignment lease
lifecycle, worker accept/progress/closeout, artifact storage, and public or
operator projection. Psionic consumes normalized closeout imports only after
those systems produce refs.

The distributed shape is:

- Psionic coordinator selects a candidate and task set.
- Local evaluator runs first for deterministic proof.
- Pylons receive independent metric-call rollout assignments through Omega.
- Rollout closeouts return normalized score, verifier, artifact, proof,
  policy, resource, route-scorecard, and failure-family refs.
- Psionic updates the candidate frontier and reflection/proposal records.

Payment modes are represented distinctly as `unpaid_smoke`, `operator_credit`,
`payable_pending_settlement`, `settled_bitcoin`, and `rejected_no_pay`.
`settled_bitcoin` is refused unless settlement receipt refs are present.
Rejected work must use `rejected_no_pay`, and `rejected_no_pay` is only valid
for rejected work.

## Failure Semantics

Rollout status separates:

- `succeeded`
- `agent_failed`
- `infrastructure_failed`
- `policy_blocked`
- `timed_out`
- `rejected`

This matters because GEPA should mutate candidate text for model/agent
failures, but infrastructure failures should not punish a candidate. Policy
violations stop advancement before rollouts run.

Live imports reject overclaims. They cannot claim public benchmark score
authority, runtime promotion to release candidate or active, or model-training
authority. Retained-smoke and validation-measured language may be retained as
metadata, but Psionic does not convert it into public leaderboard authority.

## Verification

Run:

```bash
cargo test -p psionic-train probe_gepa_rollout_coordinator --lib
cargo run -q -p psionic-train --example probe_gepa_live_closeout_import
```

The focused tests cover local Stage 0 metric calls, resumable cache behavior,
infrastructure versus agent failure classification, policy-blocked candidate
refusal, accepted and rejected live closeout imports, missing evidence-ref
rejection, settlement receipt enforcement, and import of the Stage 0 SHC Harbor
live smoke recorded by OpenAgents. They also cover the live Omega/Pylon
Terminal-Bench canary import, including its no-score, no-training, no-paid-work,
and no-settlement boundaries.
