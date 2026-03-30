# Psion Executor Long-Run Rehearsal

> Status: canonical `PSION-0606` / `#752` record, updated 2026-03-30 after
> landing the first retained long-run recovery plus replacement rehearsal for
> the admitted executor lane.

This document records the first canonical long-run rehearsal packet for the
admitted executor lane. It binds pre-flight admission, transient-interruption
recovery, export inspection, replacement validation, and review logging into
one retained result before broader long-run operating claims are trusted.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_long_run_rehearsal_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_long_run_rehearsal_fixtures
```

## What Landed

`psionic-train` now owns one typed long-run rehearsal packet that binds:

- the phase-two pre-flight checklist packet
- the continue-vs-restart incident-policy packet
- the retained 4080 interruption-recovery packet
- the retained Mac export-inspection packet
- the unified training-plus-serving throughput packet
- the canonical weekly review workflow

That means the admitted executor lane now has one machine-readable rehearsal
receipt for:

- entering the run under the frozen pre-flight contract
- surviving a transient interruption
- recovering under the admitted policy
- exporting the candidate cleanly
- validating replacement cleanly
- logging the result into the canonical review path

## Current Retained Truth

- packet digest:
  `50ebd9cad8abd98704488103686b6c21aa78f5278e85b8bb6dc19ee5666238d2`
- rehearsal id:
  `psion_executor_long_run_rehearsal_v1`
- run type:
  `cuda_4080_decision_grade`
- run id:
  `tailrun-home-admitted-20260328k`
- pre-flight digest:
  `5c8f4c8448b396b5bc1b7def7e6216cf6e03f4e2ad61e363c0d83b68faec7068`
- incident-policy digest:
  `40ef44b2c92651ab7e5ae3c7c039388bf472ca05467fc679054ffe4fb8413186`
- interruption-recovery digest:
  `d07f14dd64ce0f66d8827a9de1c6353dd5f1d001a9c81bc74669bc12e2def2c6`
- export-inspection digest:
  `9d6a39d78400f4a0c6c86398b677b9880080e8351653b3f68ccadb6e4a06aa8a`
- unified-throughput digest:
  `ff12ece15c7917e2c430cb139d81c36c2d9e2964f9ee8197275664314fc037a7`
- review-workflow digest:
  `c11b48bb9cb4381ccba810b5c154ffad6014c3b130c539a32b43dff4298078bf`
- checkpoint ref:
  `checkpoint://swarm/first-swarm-live-plan/policy`
- checkpoint step:
  `12`
- incident class:
  `transient_interruption`
- recovery action:
  `continue_from_last_green_checkpoint`
- replacement candidate row:
  `psion_executor_local_cluster_ledger_row_mlx_v1`
- rehearsal green:
  `true`

## Retained Rehearsal Rows

- `preflight_contract_green`
- `interruption_survival_green`
- `recovery_policy_green`
- `export_candidate_green`
- `replacement_validation_green`
- `review_log_green`

## Retained Review Log

- review id:
  `psion_executor_long_run_rehearsal_review_2026w14_v1`
- workflow id:
  `psion_executor_local_cluster_review_workflow_v1`
- review kind:
  `long_run_rehearsal`
- reviewer role:
  `review_cadence_owner`
- status:
  `logged_clean_rehearsal`

## Honest Current Meaning

This packet does not claim that every future long run is solved.

It does prove something narrower and useful:

- the admitted 4080 decision-grade lane now has one retained rehearsal where
  the interruption path, the policy path, the export path, the replacement
  path, and the review path are all tied together
- recovery remains governed by the same bounded transient-interruption policy
  instead of improvised operator behavior
- replacement remains blocked by serving-throughput truth if that ever
  regresses, but the retained rehearsal is green today because the unified
  throughput gate is also green

That is the closeout point: the long-run operating story is now rehearsed once
as retained machine truth instead of being spread across separate receipts with
no single closure packet.

## Follow-On Surfaces

The pre-flight packet used by this rehearsal lives at:

- `docs/PSION_EXECUTOR_PHASE_TWO_PREFLIGHT_CHECKLIST.md`
- `fixtures/psion/executor/psion_executor_phase_two_preflight_checklist_v1.json`

The incident-policy packet used by this rehearsal lives at:

- `docs/PSION_EXECUTOR_CONTINUE_RESTART_POLICY.md`
- `fixtures/psion/executor/psion_executor_continue_restart_policy_v1.json`

The interruption-recovery packet used by this rehearsal lives at:

- `docs/PSION_EXECUTOR_4080_INTERRUPTION_RECOVERY.md`
- `fixtures/psion/executor/psion_executor_4080_interruption_recovery_v1.json`

The unified-throughput packet used by this rehearsal lives at:

- `docs/PSION_EXECUTOR_UNIFIED_THROUGHPUT_REPORTING.md`
- `fixtures/psion/executor/psion_executor_unified_throughput_reporting_v1.json`

## Validation

- `cargo run -q -p psionic-train --example psion_executor_long_run_rehearsal_fixtures`
- `cargo test -q -p psionic-train psion_executor_long_run_rehearsal -- --nocapture`
