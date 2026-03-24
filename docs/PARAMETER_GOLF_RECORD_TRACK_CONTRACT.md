# Psionic Parameter Golf Record-Track Contract

> Status: canonical `PGOLF-403` / `#174` blocked record-track contract,
> updated 2026-03-23 after refreshing the committed report builder in
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

The coupled export-surface judgment now also lives at:

- `fixtures/parameter_golf/reports/parameter_golf_record_export_surface_contract.json`

## Current Contract

The record-track contract now binds together:

- the canonical acceptance report
- the landed non-record submission package surface
- the explicit record export-surface contract judgment against the README
  `train_gpt.py` surface
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
- the current export surface judgment is explicit and machine-readable
- folder-local replay verification is landed
- maintainer-facing promotion receipt and PR bundle generation are landed
- local challenge-clone dry run is landed

## Remaining Blockers

The report now keeps one blocker explicit today:

- reproducible challenge-speed `8xH100` execution is still not green

The shipped runtime-byte and build-dependency story is now explicit for the
current exported payload through the non-record accounting receipt and the
coupled research, export-surface, or record-track reports. The current export
surface is also no longer implicit: the repo now says directly that the folder
is not literally README-faithful `train_gpt.py`-only code, and instead binds a
maintainer-facing launcher-plus-runtime equivalence argument to exact shipped
bytes. The contract remains useful because it still does **not** let the repo
pretend that packaging explicitness equals real hardware closure.

## Current Honest Boundary

The strongest honest claim posture remains `non_record_submission`.

This issue retires the local counted-runtime blocker, but it does not turn
`record-track-readiness` green.

It turns the blocked state into canonical machine-readable truth so later work
can retire named blockers instead of relitigating what the blockers are.
