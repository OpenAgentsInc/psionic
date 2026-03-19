# Tassadar General Internal-Compute Red-Team Audit

Date: 2026-03-19

## Scope

This audit red-teams the public and operator-facing internal-compute boundary
surfaces that sit above the bounded execution substrate. The goal is not to
claim a wider capability. The goal is to prove that current widening work still
fails closed when someone tries to over-read named profiles as a generic broad
compute lane.

Canonical machine-readable artifacts:

- `fixtures/tassadar/reports/tassadar_general_internal_compute_red_team_route_exercises_report.json`
- `fixtures/tassadar/reports/tassadar_general_internal_compute_red_team_audit_report.json`
- `fixtures/tassadar/reports/tassadar_general_internal_compute_red_team_summary.json`

## What was red-teamed

- candidate-only broad internal-compute profiles attempting to skip their
  challenge window and route as accepted-outcome-ready
- operator-only proposal families attempting to inherit public route posture
- research-only threads and shared-state profiles attempting to widen into
  public publication
- relaxed-SIMD attempting to inherit deterministic SIMD publication
- broad claim language attempting to over-read the current lane as arbitrary
  Wasm or silent broad internal compute

## Current result

The current red-team audit is clean.

- route exercises stay blocked on all audited cases
- the effective-unbounded claim checker still names arbitrary Wasm execution as
  out of scope
- the frozen core-Wasm public acceptance gate still suppresses public closure
- relaxed-SIMD remains explicitly non-promoted
- shared-state concurrency remains operator-bounded with public suppression

## Why this matters

The broader internal-compute lane now has an explicit disclosure-safe audit
that tests failure modes at the publication boundary rather than only technical
success paths. That gives Psionic one honest statement it can make:

`The broader internal-compute lane has been red-teamed successfully, and the
current named-profile, proposal-family, and concurrency boundaries still fail
closed.`

It still does not imply:

- arbitrary Wasm execution
- silent proposal-family inheritance
- public threads publication
- default served relaxed-SIMD

## Follow-on

This audit is the governance pre-closeout predecessor for the pre-closeout
universality claim-boundary report. The next step is to freeze exactly what is
true before the terminal universal-substrate contract starts, not to widen the
current public claim.
