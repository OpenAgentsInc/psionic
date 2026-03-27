# HOMEGOLF Clustered Run Surface Audit

Date: March 27, 2026

## Summary

This audit records the first honest clustered HOMEGOLF score surface.

Retained machine-readable report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`

Generator and checker:

- `crates/psionic-train/src/parameter_golf_homegolf_clustered.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_clustered_run_surface.rs`
- `scripts/check-parameter-golf-homegolf-clustered-run-surface.sh`

## What Landed

The repo now freezes one explicit HOMEGOLF report that binds together two
already-retained truths:

- one real admitted-device home-Tailnet run:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json`
- one real exact-family HOMEGOLF train-to-infer proof:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json`

This means HOMEGOLF now has one machine-readable clustered score surface
instead of leaving operators to infer a relationship between separate swarm and
exact-family reports.

## Retained Values

The retained clustered surface keeps:

- admitted device set:
  - `local_m5_mlx`
  - `archlinux_rtx4080_cuda`
- cluster wallclock under the HOMEGOLF cap:
  - `observed_cluster_wallclock_ms=61228`
  - `wallclock_cap_seconds=600`
- merged dispositions from the admitted-device run:
  - `merge=merged`
  - `publish=refused`
  - `promotion=held`
- exact-family bundle proof bindings:
  - `descriptor_digest=8a111f908acf02174554a75e83e13092852ee7caa534f65c43be897bd4c606ee`
  - `tokenizer_digest=49b44264442058c20b2b95a947f3aac60e8729fd57d63db8b8754de8edb98a6d`
- exact-family score and bundle size:
  - `final_validation_mean_loss=8.60598874092102`
  - `final_validation_bits_per_byte=9.93265382277841`
  - `model_artifact_bytes=68248296`
- infer/serve closure on the retained prompt:
  - prompt `abcd`
  - direct tokens `[7, 2, 7, 2]`
  - served tokens `[7, 2, 7, 2]`

## Why This Surface Exists

Before this landed, HOMEGOLF had two incomplete but real ingredients:

- real clustered-device contribution truth on the home Tailnet
- real exact-family train-to-infer closure for the HOMEGOLF-compatible `9x512`
  family

What it did not have was one report that put those truths in the same place
with an explicit claim boundary.

That gap mattered because operators could otherwise overread the repo and claim
that one exact dense mixed-device home-cluster run already emitted the scored
bundle directly.

The new clustered surface closes that documentation and reporting gap without
making that false claim.

## What This Proves

The repo can now truthfully say:

- one real two-device home-Tailnet run completed inside the HOMEGOLF `600s`
  budget posture
- the exact admitted device inventory and per-device contribution receipts are
  retained inside one HOMEGOLF report
- that same report binds one exact-family HOMEGOLF bundle proof and final
  `val_bpb`
- the current HOMEGOLF track now has a first clustered score surface instead of
  only separate swarm and exact-family proofs

## What This Does Not Prove

This surface still does **not** prove:

- that one live exact dense mixed-device home-cluster run already produced the
  scored bundle directly
- that the current exact dense HOMEGOLF trainer already executes on the M5 plus
  RTX 4080 home pair
- that the current retained `val_bpb` is a public-leaderboard-equivalent score
- that the current retained model artifact is inside the contest-size budget

That remaining work belongs to later HOMEGOLF issues, not to this audit.

## Verification

The landed surface was revalidated with:

```bash
cargo run -q -p psionic-train --bin parameter_golf_homegolf_clustered_run_surface -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json

cargo test -q -p psionic-train clustered_homegolf_surface -- --nocapture

./scripts/check-parameter-golf-homegolf-clustered-run-surface.sh
```

Those checks verify that the committed report still matches the retained
Tailnet admitted-home summary and the retained exact-family HOMEGOLF bundle
proof.
