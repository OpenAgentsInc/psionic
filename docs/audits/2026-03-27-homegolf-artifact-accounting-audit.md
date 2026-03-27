# HOMEGOLF Artifact Accounting Audit

Date: March 27, 2026

## Summary

This audit records the upgraded counted-byte answer for the HOMEGOLF track.

Retained machine-readable report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_artifact_accounting.json`

Generator and checker:

- `crates/psionic-train/src/parameter_golf_homegolf_accounting.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_artifact_accounting.rs`
- `scripts/check-parameter-golf-homegolf-artifact-accounting.sh`

## Source Surfaces

The accounting report binds two retained sources:

- scored HOMEGOLF clustered surface:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`
- current Psionic counted-code posture:
  `fixtures/parameter_golf/reports/parameter_golf_research_harness_report.json`

From those sources, the report keeps:

- `merged_bundle_descriptor_digest=af7583b983cd2016d1c4aa7b8e185557048539f3be50db26873247e4d2bc9981`
- `merged_bundle_tokenizer_digest=4f5e8adb109c66b4886963bc75a7befd73bda36d27fd7102df8e9e66503b0e2a`
- `counted_code_bytes=7188700`
- `scored_model_artifact_bytes=4732744`

## Retained Budget Result

The report computes:

- `total_counted_bytes=11921444`
- `artifact_cap_bytes=16000000`
- `cap_delta_bytes=-4078556`
- `budget_status=within_artifact_cap`

So the current honest HOMEGOLF budget answer is simple:

- the retained counted posture is under the cap by `4,078,556` bytes
- the current result is a pass, not a refusal

## What This Proves

The repo can now truthfully say:

- HOMEGOLF has one explicit counted-byte report bound to the live dense
  mixed-device surface
- the current counted code and scored model bytes are preserved separately
- the total counted bytes and cap delta are machine-readable
- the current byte-budget outcome is explicit and within the contest cap

## What This Does Not Prove

This report still does **not** prove:

- that HOMEGOLF is already contest-ready
- that the current counted-code posture is the final optimized Psionic export
  surface
- that later HOMEGOLF packaging work cannot reduce the code-byte side
- that later HOMEGOLF training/export work cannot reduce the model-byte side

It intentionally records today's real answer so later work can improve quality
and runtime without losing the byte-budget pass.

## Verification

The landed accounting surface was revalidated with:

```bash
cargo run -q -p psionic-train --bin parameter_golf_homegolf_artifact_accounting -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_artifact_accounting.json

cargo test -q -p psionic-train homegolf_artifact_accounting -- --nocapture

./scripts/check-parameter-golf-homegolf-artifact-accounting.sh
```
