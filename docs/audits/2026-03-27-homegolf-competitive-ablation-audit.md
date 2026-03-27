# HOMEGOLF Competitive Ablation Audit

Date: 2026-03-27

## Goal

Close `HOMEGOLF-11` by making the exact HOMEGOLF lane admit at least one
competitive variant beyond the naive baseline and by retaining one explicit
machine-readable report of which public-winning Parameter Golf techniques are
now wired into that exact lane.

## What Changed

The exact HOMEGOLF dense trainer is no longer hard-wired to only the naive
`baseline_sp1024_9x512` family shape.

The canonical exact trainer now admits:

- `baseline_sp1024_9x512`
- `competitive_homegolf_v1`

`competitive_homegolf_v1` is built only from already-owned Psionic model and
trainer surfaces. It does not invent a detached contest-only stack. The exact
lane now wires these surfaces into one retained best-known competitive
configuration:

- `BigramHash`
- partial RoPE
- deep-layer XSA
- `LeakyReLU(0.5)^2`
- late-layer value embeddings
- EMA
- SWA sourced from EMA
- legal score-first TTT
- competitive final artifact defaults

## Exact Surfaces Landed

- model-family constructor:
  `crates/psionic-models/src/parameter_golf.rs`
- exact trainer variant admission:
  `crates/psionic-train/src/parameter_golf_single_h100_training.rs`
- exact trainer CLI variant selector:
  `crates/psionic-train/src/bin/parameter_golf_single_h100_train.rs`
- retained ablation generator:
  `crates/psionic-train/src/parameter_golf_homegolf_competitive_ablation.rs`
- retained ablation entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_competitive_ablation.rs`
- retained report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_competitive_ablation.json`
- checker:
  `scripts/check-parameter-golf-homegolf-competitive-ablation.sh`
- canonical track doc:
  `docs/HOMEGOLF_TRACK.md`
- canonical track contract:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_track_contract.json`

## Retained Best-Known Variant

The retained report freezes this exact-lane best-known variant:

- model variant: `competitive_homegolf_v1`
- validation mode: `sliding_window:64`
- score-first TTT: enabled
- EMA: enabled
- SWA: enabled, sourced from EMA
- final model surface: `swa`
- final artifact config: `competitive_defaults()`

The exact dense training command template retained in the report is:

```sh
PSIONIC_PARAMETER_GOLF_MODEL_VARIANT=competitive_homegolf_v1 cargo run -q -p psionic-train --bin parameter_golf_single_h100_train -- ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model <training_report_path> both sliding_window:64 score_first_ttt
```

This keeps the exact HOMEGOLF posture while replacing the old naive bounded
proof shape as the best-known exact-lane configuration.

## Explicit Refusals

The retained ablation report also keeps explicit refusals for techniques that
are still outside the exact HOMEGOLF lane today:

- `smeargate`
  Reason: not yet exposed by the current promoted PGOLF family contract.
- `mixed_qat_train_time`
  Reason: the exact trainer owns competitive export surfaces, but not yet one
  train-time mixed-bit QAT contract.

This is important because `HOMEGOLF-11` is not “all public tricks copied
blindly.” It is “all admitted useful tricks assembled into one exact lane, and
the remaining ones called out honestly.”

## Machine-Readable Result

Retained report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_competitive_ablation.json`

Key retained facts:

- `report_id=parameter_golf.homegolf_competitive_ablation.v1`
- `best_known_variant_replaces_bounded_proof_baseline=true`
- `best_known_lane.model_variant=competitive_homegolf_v1`
- `report_digest=9dc82dddcb368fa11fb921de6456b9dddb5f0f25d43d5f2d2eb226d6a3c3f55a`

## Validation Run

I validated the landed surface with:

```sh
cargo run -q -p psionic-train --bin parameter_golf_homegolf_track_contract -- fixtures/parameter_golf/reports/parameter_golf_homegolf_track_contract.json
cargo run -q -p psionic-train --bin parameter_golf_homegolf_competitive_ablation -- fixtures/parameter_golf/reports/parameter_golf_homegolf_competitive_ablation.json
cargo test -q -p psionic-models competitive_homegolf_v1_enables_competitive_family_surfaces -- --nocapture
cargo test -q -p psionic-train competitive_ablation_keeps_best_known_competitive_variant -- --nocapture
cargo test -q -p psionic-train competitive_homegolf_defaults_enable_competitive_surfaces -- --nocapture
cargo test -q -p psionic-train model_variant_parse_accepts_competitive_label -- --nocapture
./scripts/check-parameter-golf-homegolf-competitive-ablation.sh
```

## Honest Boundary

`HOMEGOLF-11` closes the exact-lane competitive wiring gap. It does not yet
claim that this best-known variant has already posted the best retained
HOMEGOLF score on local mixed hardware. That remains later work for the live
dense HOMEGOLF runtime and scorepath issues.
