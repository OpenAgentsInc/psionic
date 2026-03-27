# HOMEGOLF Multi-Seed Package Audit

Date: March 27, 2026

## Summary

This audit records the first retained repeated-run HOMEGOLF package.

Retained machine-readable package:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_multiseed_package.json`

Retained repeated run receipts:

- `fixtures/parameter_golf/reports/homegolf_multiseed/parameter_golf_homegolf_dense_bundle_proof_seed_000.json`
- `fixtures/parameter_golf/reports/homegolf_multiseed/parameter_golf_homegolf_dense_bundle_proof_seed_001.json`
- `fixtures/parameter_golf/reports/homegolf_multiseed/parameter_golf_homegolf_dense_bundle_proof_seed_002.json`

Generator and checker:

- `crates/psionic-train/src/parameter_golf_homegolf_multiseed_package.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_multiseed_package.rs`
- `scripts/check-parameter-golf-homegolf-multiseed-package.sh`

## What Landed

`HOMEGOLF-14` does not pretend that the current lane is already
baseline-competitive. Instead it freezes the minimum honest repeated-run bar
for the current deterministic exact-family HOMEGOLF proof lane:

- three retained repeated receipts
- exact per-run bundle-proof metrics
- mean and spread over `val_bpb`
- explicit mean deltas versus the public naive baseline and current public best
- explicit refusal of stronger “beat the public reference” language

## Retained Repeated Result

The current repeated package keeps:

- repeated run count: `3`
- claim class: `public_baseline_comparable_only`
- artifact budget status: `within_artifact_cap`
- mean `val_bpb`: `9.93265382277841`
- stddev `val_bpb`: `0.0`
- min `val_bpb`: `9.93265382277841`
- max `val_bpb`: `9.93265382277841`
- mean delta versus public naive baseline: `8.70828812277841`
- mean delta versus current public best: `8.813253822778409`

Each repeated receipt also preserves:

- exact run id
- exact descriptor digest
- exact tokenizer digest
- exact model bytes
- exact direct-versus-served parity

## Why Zero Spread Is Honest Here

The current HOMEGOLF repeated package is built over the repo-owned bounded
exact-family proof lane. That lane is deterministic today, so repeated runs are
expected to reproduce the same result exactly.

That means:

- zero spread is not fake variance suppression
- zero spread is itself useful evidence that the current proof lane is
  reproducible
- stronger statistical language is still refused because reproducibility alone
  is not the same thing as competitiveness

## What This Proves

The repo can now truthfully say:

- HOMEGOLF has a retained repeated-run result package
- that package preserves exact per-run receipts instead of only one canonical
  proof
- the package computes mean and spread for `val_bpb`
- the package keeps explicit mean deltas against the public naive baseline and
  the current public best
- the package keeps a stronger-claim bar instead of letting repeated runs
  automatically justify contest rhetoric

## What This Does Not Prove

This package still does **not** prove:

- that HOMEGOLF is baseline-competitive
- that HOMEGOLF can honestly claim to beat the public naive baseline
- that HOMEGOLF can honestly claim to beat the current public best
- that the repeated package is already a live dense mixed-device multi-seed
  contest package

The current package is reproducibility-grade evidence for the exact-family
proof lane, not a win claim.

## Verification

The package surface was revalidated with:

```bash
cargo build -q -p psionic-serve --example parameter_golf_homegolf_dense_bundle_proof

target/debug/examples/parameter_golf_homegolf_dense_bundle_proof \
  /tmp/homegolf_multiseed_seed_000 0

target/debug/examples/parameter_golf_homegolf_dense_bundle_proof \
  /tmp/homegolf_multiseed_seed_001 1

target/debug/examples/parameter_golf_homegolf_dense_bundle_proof \
  /tmp/homegolf_multiseed_seed_002 2

cargo run -q -p psionic-train --bin parameter_golf_homegolf_multiseed_package -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_multiseed_package.json

./scripts/check-parameter-golf-homegolf-multiseed-package.sh
```
