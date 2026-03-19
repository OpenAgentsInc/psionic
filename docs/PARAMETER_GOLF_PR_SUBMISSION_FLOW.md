# Psionic Parameter Golf PR Submission Flow

> Status: canonical `PGOLF-701` / `#191`, `PGOLF-702` / `#192`, and
> `PGOLF-703` / `#193` promotion, PR-bundle, and local-clone dry-run contract,
> updated 2026-03-18 after landing
> `crates/psionic-eval/src/parameter_golf_promotion.rs` and
> `crates/psionic-train/src/parameter_golf_submission_pr.rs`.

This document records the maintainer-facing gate Psionic now applies before
calling a Parameter Golf folder PR-ready.

## What Landed

`psionic-eval` now exposes:

- `ParameterGolfSubmissionPromotionCandidate`
- `ParameterGolfSubmissionPromotionReceipt`
- `build_parameter_golf_submission_promotion_receipt(...)`

`psionic-train` now exposes:

- `ParameterGolfFinalPrBundleReport`
- `ParameterGolfLocalCloneDryRunReport`
- `write_parameter_golf_final_pr_bundle(...)`
- `write_parameter_golf_final_pr_bundle_report(...)`
- `write_parameter_golf_local_clone_dry_run_report(...)`

The repo now also carries:

- `fixtures/parameter_golf/reports/parameter_golf_final_pr_bundle.json`
- `fixtures/parameter_golf/reports/parameter_golf_local_clone_dry_run.json`

## Promotion Receipt

The promotion receipt now makes the public README gate machine-readable:

- the current best record-track baseline being compared against
- the candidate delta in `bits per byte` and `nats per byte`
- whether the README's `0.005` nat threshold is cleared
- whether significance evidence is present when required
- whether the systems-only waiver is claimed and actually supported
- whether the supplied evidence supports a promotion claim

That means Psionic no longer has to leave record-promotion posture as prose in
README notes or issue comments.

## Final PR Bundle

The final PR-bundle generator now emits one deterministic output root
containing:

- the exact `records/.../<submission_id>` folder
- the exported-folder run-evidence report
- the exported-folder replay-verification report
- the promotion receipt
- maintainer-facing checklist text for the live challenge repo

The canonical generator pins the bounded local-reference wallclock receipt to
the committed reference measurement so the non-record review bundle digest
stays stable across reruns.

Generate it with:

```bash
cargo run -p psionic-train --example parameter_golf_final_pr_bundle \
  /tmp/psionic_parameter_golf_final_pr_bundle
```

## Local Challenge-Clone Dry Run

The last anti-drift gate is now also explicit.

Run the full local-clone dry run with:

```bash
cargo run -p psionic-train --example parameter_golf_local_clone_dry_run \
  ~/code/parameter-golf \
  /tmp/parameter_golf_local_clone_dry_run.json
```

That flow now:

- generates the final PR bundle
- stages the folder into the live local `parameter-golf` clone
- re-runs the compatibility verifier there
- re-runs the folder-local replay verifier there
- removes the staged folder and confirms the clone returns to its original
  `git status --short --branch` state

## Current Honest Boundary

This closes the PR mechanics only.

What is now explicit:

- promotion posture is typed rather than implied
- one exact PR bundle exists
- one local challenge-clone dry run is preserved

What is still not implied:

- record-track readiness
- CUDA baseline closure
- true measured `8xH100` success

The current promotion receipt remains a refusal for the bounded non-record
bundle. That is the intended claim boundary until stronger record evidence
actually exists.
