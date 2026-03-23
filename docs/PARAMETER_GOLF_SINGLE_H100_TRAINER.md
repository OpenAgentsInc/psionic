# Psionic Parameter Golf Single-H100 Trainer

> Status: canonical bounded Rust-native Parameter Golf single-H100 trainer
> command, written 2026-03-23 after landing
> `crates/psionic-train/src/parameter_golf_single_h100_training.rs` and
> `crates/psionic-train/src/bin/parameter_golf_single_h100_train.rs`.

This document records the current honest trainer posture for the Psionic
Parameter Golf single-H100 lane.

It is narrower than a Google lane, `8xH100`, or record-track claim.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfSingleH100TrainingConfig`
- `ParameterGolfSingleH100TrainingDisposition`
- `ParameterGolfSingleH100TrainingReport`
- `build_parameter_golf_single_h100_training_report(...)`
- `write_parameter_golf_single_h100_training_report(...)`
- `parameter_golf_single_h100_train` as a Rust binary entrypoint

That means the repo now owns one real Rust entrypoint for the public
single-device baseline path rather than only the narrower bring-up seam and the
bounded local-reference trainer.

## Command

The binary defaults to the local `~/code/parameter-golf` clone paths from the
public README:

```bash
cargo run -q -p psionic-train --bin parameter_golf_single_h100_train
```

You can also pass the explicit cached-dataset and tokenizer paths, an output
report path, and a bounded step count:

```bash
cargo run -q -p psionic-train --bin parameter_golf_single_h100_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/parameter_golf_single_h100_training.json \
  1
```

The first proof path intentionally keeps `max_steps=1` while preserving the
real challenge single-device geometry and final validation contract.

## Data Setup

This doc assumes the public challenge cache has already been downloaded into
the local `~/code/parameter-golf` clone using the public README workflow:

```bash
cd ~/code/parameter-golf
.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024
```

On the current local machine, that populates:

- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/fineweb_1024_bpe.model`
- `data/tokenizers/fineweb_1024_bpe.vocab`

## What The Command Binds

The command is explicit about what it treats as trainer truth. It binds:

- the local cached FineWeb `sp1024` shard directory into a versioned
  `DatasetManifest`
- the local tokenizer file into a machine-readable `TokenizerDigest`
- the local CUDA inventory into explicit single-H100 machine-admission truth
  using the repo-owned CUDA discovery substrate
- the public single-device batch geometry from
  `ParameterGolfBatchGeometry::challenge_single_device_defaults()`
- the public baseline `9x512` model contract and optimizer-plan digest
- one bounded optimizer-step run on CUDA when the machine and CUDA-capability
  contracts are satisfied
- final validation loss and `val_bpb` directly from the Psionic path
- post-step int8-plus-zlib artifact bytes, artifact ref, and artifact digest
- per-step and aggregate phase timings so later profiling work can reuse the
  same trainer receipt

The report also preserves explicit refusal when:

- the dataset root is missing or malformed
- the tokenizer path is missing
- the machine contract is not one qualifying non-MIG H100
- the committed Parameter Golf CUDA capability report still carries trainer
  blockers

## Current Honest Boundary

This command is still a bounded first single-H100 proof rather than a stronger
contest claim.

Today the single-H100 trainer doc does **not** claim:

- a Google operator lane
- `8xH100` distributed closure
- leaderboard-speed runtime closure
- record-track accounting closure

Instead, it gives the repo one narrower but important thing:

- a real Rust-owned single-H100 baseline training command that binds the
  challenge dataset, tokenizer, machine contract, final validation, and
  compressed-model accounting surfaces into one machine-readable report

The narrower machine-admission seam from
`docs/PARAMETER_GOLF_SINGLE_H100_BRINGUP.md` remains useful, but it is no
longer the only repo-owned single-H100 entrypoint.
