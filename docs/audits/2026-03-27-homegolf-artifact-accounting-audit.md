# HOMEGOLF Artifact Accounting Audit

Date: March 27, 2026

## Summary

This audit records the first explicit counted-byte answer for the HOMEGOLF
track.

Retained machine-readable report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_artifact_accounting.json`

Generator and checker:

- `crates/psionic-train/src/parameter_golf_homegolf_accounting.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_artifact_accounting.rs`
- `scripts/check-parameter-golf-homegolf-artifact-accounting.sh`

## Source Surfaces

The accounting report binds two existing retained sources:

- scored HOMEGOLF clustered surface:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`
- current Psionic counted-code posture:
  `fixtures/parameter_golf/reports/parameter_golf_research_harness_report.json`

From those sources, the report keeps:

- `merged_bundle_descriptor_digest=8a111f908acf02174554a75e83e13092852ee7caa534f65c43be897bd4c606ee`
- `merged_bundle_tokenizer_digest=49b44264442058c20b2b95a947f3aac60e8729fd57d63db8b8754de8edb98a6d`
- `counted_code_bytes=7188700`
- `scored_model_artifact_bytes=68248296`

## Retained Budget Result

The report computes:

- `total_counted_bytes=75436996`
- `artifact_cap_bytes=16000000`
- `cap_delta_bytes=59436996`
- `budget_status=refused_exceeds_artifact_cap`

So the current honest HOMEGOLF budget answer is simple:

- the scored exact-family bundle is over the cap by `59,436,996` bytes
- the current result is a refusal, not a pass

## What This Proves

The repo can now truthfully say:

- HOMEGOLF has one explicit counted-byte report bound to the exact scored
  clustered surface
- the current counted code and scored model bytes are preserved separately
- the total counted bytes and cap delta are machine-readable
- the current byte-budget outcome is explicit instead of being implied or left
  for a human to reconstruct

## What This Does Not Prove

This report still does **not** prove:

- that HOMEGOLF is already contest-ready
- that the current counted-code posture is the final optimized Psionic export
  surface
- that later HOMEGOLF packaging work cannot reduce the code-byte side
- that later HOMEGOLF training/export work cannot reduce the model-byte side

It intentionally records today's real answer so later work has a fixed baseline
to improve from.

## Verification

The landed accounting surface was revalidated with:

```bash
cargo run -q -p psionic-train --bin parameter_golf_homegolf_artifact_accounting -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_artifact_accounting.json

cargo test -q -p psionic-train homegolf_artifact_accounting -- --nocapture

./scripts/check-parameter-golf-homegolf-artifact-accounting.sh
```
