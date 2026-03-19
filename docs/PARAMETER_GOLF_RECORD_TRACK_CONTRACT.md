# Psionic Parameter Golf Record-Track Contract

> Status: canonical `PGOLF-403` / `#174` blocked record-track contract,
> updated 2026-03-18 after landing the committed report builder in
> `crates/psionic-train/src/parameter_golf_record_track.rs`.

This document records the current record-track posture for the Psionic
Parameter Golf lane.

It is intentionally a blocked contract, not a green readiness claim.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfRecordTrackContractReport`
- `build_parameter_golf_record_track_contract_report()`
- `write_parameter_golf_record_track_contract_report(...)`

The committed machine-readable report now lives at:

- `fixtures/parameter_golf/reports/parameter_golf_record_track_contract.json`

## Current Contract

The record-track contract now binds together:

- the canonical acceptance report
- the landed non-record submission package surface
- the exported-folder evidence and replay-verification reports
- the promotion receipt plus final PR-bundle and local challenge-clone dry-run reports
- the committed post-parity research harness
- the distributed `8xH100` benchmark reference

That means the repo now has one explicit place where the record-track blockers
are named instead of implied.

## Satisfied Surfaces

The current contract now keeps these satisfied surfaces explicit:

- challenge-oracle parity is landed
- reproducible record-folder output exists for the non-record lane
- counted-byte vocabulary is explicit and machine-readable
- folder-local replay verification is landed
- maintainer-facing promotion receipt and PR bundle generation are landed
- local challenge-clone dry run is landed

## Remaining Blockers

The report keeps two blockers explicit today:

- the shipped `train_gpt.py` is now a real non-record launcher, but no
  defended counted-runtime or build-dependency story exists yet for a true
  record-track execution payload
- reproducible challenge-speed `8xH100` execution is still not green

So the contract is useful precisely because it does **not** let the repo
pretend that packaging explicitness equals record readiness.

## Current Honest Boundary

The strongest honest claim posture remains `non_record_submission`.

This issue does not turn `record-track-readiness` green.

It turns the blocked state into canonical machine-readable truth so later work
can retire named blockers instead of relitigating what the blockers are.
