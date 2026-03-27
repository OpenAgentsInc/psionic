# HOMEGOLF Public Comparison Audit

Date: March 27, 2026

## Summary

This audit records the first deterministic HOMEGOLF comparison report against
the public Parameter Golf baseline and the current public best leaderboard row
from the reviewed repo snapshot.

Retained machine-readable report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_public_comparison.json`

Generator and checker:

- `crates/psionic-train/src/parameter_golf_homegolf_comparison.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_public_comparison.rs`
- `scripts/check-parameter-golf-homegolf-public-comparison.sh`

## Public Reference Snapshot

The comparison report freezes these public references from the reviewed local
Parameter Golf repo snapshot:

- naive baseline:
  `competition/repos/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json`
  - `val_bpb=1.2243657`
  - `artifact_bytes=15863489`
  - `wallclock_cap_seconds=600`
- current public leaderboard best:
  `competition/repos/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/{submission.json,README.md}`
  - `val_bpb=1.1194`
  - `artifact_bytes=15990006`
  - `wallclock_cap_seconds=600`

These numbers are intentionally frozen into Psionic's report generator so the
committed comparison fixture stays deterministic even if the sibling
competition clone changes later.

## HOMEGOLF Side Bound Into The Report

The HOMEGOLF side comes from:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`

The current retained HOMEGOLF values are:

- `final_validation_bits_per_byte=9.93265382277841`
- `homegolf_scored_artifact_bytes=68248296`
- `homegolf_wallclock_cap_seconds=600`
- `homegolf_observed_cluster_wallclock_ms=61228`

## Retained Deltas

Against the public naive baseline:

- `delta_val_bpb=8.70828812277841`
- `delta_artifact_bytes=52384807`
- `delta_wallclock_cap_seconds=0`

Against the current public leaderboard best:

- `delta_val_bpb=8.81325382277841`
- `delta_artifact_bytes=52258290`
- `delta_wallclock_cap_seconds=0`

## What This Proves

The repo can now truthfully say:

- HOMEGOLF emits one explicit comparison artifact beside the retained clustered
  score surface
- the comparison artifact records exact deltas against the public naive
  baseline and the current public best leaderboard row
- the allowed comparison language is now frozen inside one machine-readable
  report:
  - `public-baseline comparable`
  - `not public-leaderboard equivalent`

## What This Does Not Prove

This report still does **not** prove:

- public leaderboard equivalence for HOMEGOLF
- full contest-style counted-artifact parity on the HOMEGOLF side
- that the HOMEGOLF result is already a valid public submission candidate

The current HOMEGOLF bytes in this comparison are the retained scored bundle
bytes from the clustered surface, not yet the later full counted-artifact
receipt that includes the HOMEGOLF-specific export/accounting posture.

That remaining accounting closure belongs to the next HOMEGOLF issue.

## Verification

The landed comparison surface was revalidated with:

```bash
cargo run -q -p psionic-train --bin parameter_golf_homegolf_public_comparison -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_public_comparison.json

cargo test -q -p psionic-train homegolf_public_comparison -- --nocapture

./scripts/check-parameter-golf-homegolf-public-comparison.sh
```
