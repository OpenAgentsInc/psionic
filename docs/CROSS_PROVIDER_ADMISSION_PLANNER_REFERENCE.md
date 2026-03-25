# Cross-Provider Admission Planner Reference

> Status: canonical `XTRAIN-13` / `#529` record, updated 2026-03-25 after
> landing the cross-provider admission planner in
> `crates/psionic-train/src/cross_provider_admission_planner.rs`.

This document records the deterministic planner that ranks current Google,
RunPod, and local sources for the admitted execution classes in the
cross-provider pretraining program.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-cross-provider-admission-planner.sh
```

## What Landed

`psionic-train` now owns one typed admission planner that freezes:

- target counts for dense ranks, contributor windows, validators,
  checkpoint writers, eval workers, and data builders
- a deterministic score breakdown over trust, network, storage, cost, backend,
  and binder alignment
- typed refusals for source-contract and planner-policy rejections
- one machine-legible placement order for every admitted execution class

The landed surface includes:

- `CrossProviderAdmissionPlan`
- `CrossProviderAdmissionRoleTarget`
- `CrossProviderAdmissionCandidateEvaluation`
- `CrossProviderAdmissionPlannerRefusalKind`
- the binary `cross_provider_admission_plan`
- the checker `scripts/check-cross-provider-admission-planner.sh`
- the committed fixture `fixtures/training/cross_provider_admission_plan_v1.json`

## Why This Matters

One cross-provider train system needs one explicit answer for why a machine was
selected, de-prioritized, or refused. This planner turns the retained machine
contracts into one deterministic role-placement surface instead of leaving
placement to runbook prose or operator instinct.

## Current Limits

This issue intentionally does not claim:

- automatic provider acquisition
- public-swarm discovery or adversarial admission
- same-job mixed-backend dense placement

