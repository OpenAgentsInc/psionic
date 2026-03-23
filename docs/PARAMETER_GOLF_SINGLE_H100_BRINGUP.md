# Psionic Parameter Golf Single-H100 Bring-Up

> Status: canonical Rust-native single-H100 bring-up seam for Parameter Golf,
> added 2026-03-18 after landing
> `crates/psionic-train/src/parameter_golf_single_h100_bringup.rs` and
> `crates/psionic-train/src/bin/parameter_golf_single_h100_bringup.rs`.
> Parameter Golf development stopped on 2026-03-19 before this seam became a
> full Rust-only trainer path; see `docs/PARAMETER_GOLF_AFTER_ACTION.md`.

Update 2026-03-23: the real bounded trainer entrypoint now exists separately in
`docs/PARAMETER_GOLF_SINGLE_H100_TRAINER.md`. This bring-up doc remains the
narrower machine-admission and first-microbatch seam.

This document records the current honest single-H100 bring-up posture for the
Psionic Parameter Golf lane.

It is intentionally narrower than a real training-readiness claim.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfSingleH100BringupConfig`
- `ParameterGolfSingleH100BringupDisposition`
- `ParameterGolfSingleH100ExecutionPosture`
- `ParameterGolfSingleH100BringupReport`
- `build_parameter_golf_single_h100_bringup_report(...)`
- `write_parameter_golf_single_h100_bringup_report(...)`
- `parameter_golf_single_h100_bringup` as a Rust binary entrypoint

The canonical machine-readable report and checker now live at:

- `fixtures/parameter_golf/reports/parameter_golf_single_h100_bringup.json`
- `scripts/check-parameter-golf-single-h100-bringup.sh`

That means the repo now owns one real Psionic-native command for the public
single-device challenge contract instead of only issue text and the exported
folder's Python launcher, and that command now preserves explicit local CUDA
machine-admission truth in the same report. When the machine contract is
satisfied, it also materializes the first real challenge training microbatch
and computes a bounded CPU-reference loss over a small leading prefix of that
exact token window.

## Command

The binary defaults to the local `~/code/parameter-golf` clone paths from the
public README:

```bash
cargo run -q -p psionic-train --bin parameter_golf_single_h100_bringup
```

You can also pass the explicit cached-dataset and tokenizer paths plus an
output report path:

```bash
cargo run -q -p psionic-train --bin parameter_golf_single_h100_bringup -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/parameter_golf_single_h100_bringup.json
```

To reproduce the current committed evidence against the local challenge cache:

```bash
scripts/check-parameter-golf-single-h100-bringup.sh
```

## Data Setup

This doc assumes the public challenge cache has already been downloaded into
the local `~/code/parameter-golf` clone using the public README workflow:

```bash
cd ~/code/parameter-golf
.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024
```

On the current local machine, that populated:

- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/fineweb_1024_bpe.model`
- `data/tokenizers/fineweb_1024_bpe.vocab`

## What The Command Binds

The command is deliberately narrow and explicit. It binds:

- the local cached FineWeb `sp1024` shard directory into a versioned
  `DatasetManifest`
- the local tokenizer file into a machine-readable `TokenizerDigest`
- the local CUDA inventory into explicit single-H100 machine-admission truth
  using the repo-owned CUDA discovery substrate
- the public single-device batch geometry from
  `ParameterGolfBatchGeometry::challenge_single_device_defaults()`
- the public baseline `9x512` model contract and optimizer-plan digest
- when local CUDA inventory satisfies the single-H100 admission contract, the
  first real challenge training microbatch and a bounded CPU-reference mean
  loss over the leading sequences from that exact token window
- the current CUDA blocker list from
  `builtin_parameter_golf_cuda_training_capability_report()`
- the observed wallclock of the bring-up command itself
- explicit absence of `val_loss`, `val_bpb`, and compressed-model bytes while
  training does not execute yet

The output report records the exact dataset root, tokenizer path, manifest
digest, local CUDA inventory, machine-admission thresholds, model descriptor
digest, optimizer-plan digest, blocker list, observed wallclock, current
training-execution posture, and optional first-microbatch probe used by the
Rust command.

## Current Committed Evidence

The current committed report captures one real local run against the downloaded
challenge cache with:

- `train_shard_count = 80`
- `validation_shard_count = 1`
- `train_token_count = 8_000_000_000`
- `validation_token_count = 62_021_846`
- `dataset_manifest_digest = e83387a68b96075c410061b62f7c82bdc86449f8b6d9890bb6fd1a61210396d4`
- `tokenizer_digest = 4f5e8adb109c66b4886963bc75a7befd73bda36d27fd7102df8e9e66503b0e2a`
- `first_observed_cuda_device = NVIDIA GeForce RTX 4080`
- `matching_h100_device_count = 0`
- `machine_contract_satisfied = false`
- `execution_posture = contract_validation_only`
- `disposition = refused_machine_contract`

That is the current honest single-H100 parity result: the command has been run
against the real public cache, but it still refuses before training because the
local review host is not a qualifying H100 machine.

On a qualifying non-MIG H100 machine, the command now becomes
`ready_to_attempt`: it materializes the exact first challenge microbatch from
the cached FineWeb `sp1024` shards and computes a bounded CPU-reference mean
loss for a small leading prefix of that window. That is still not a CUDA
training claim.

## Current Honest Boundary

This is not yet a successful Rust-only training run.

Today the single-H100 bring-up command does **not** claim:

- that Psionic already owns the full CUDA training path for the public baseline
- that the repo can already execute the real baseline train loop on one H100
- that the later `8xH100` record-attempt lane is ready

Instead, it does one narrower but important job:

- it gives the repo a real Rust-native entrypoint for the public single-H100
  contract
- it preserves one committed report from the real public cache plus a checker
  that revalidates the invariant contract against the same cache
- it preserves explicit local machine refusal when the host is not a usable
  H100 target instead of pretending the cache alone is enough
- on a qualifying H100, it proves the Rust path can materialize the first real
  challenge microbatch and evaluate a bounded prefix of that microbatch with
  the CPU reference model
- it preserves the exact dataset or tokenizer or model or blocker truth that
  later work must reuse instead of rebuilding from memory

This remains the historical bridge that had been intended for `PGOLF-604`,
`PGOLF-605`, and the existing `PGOLF-601` and `PGOLF-602` work before the lane
stopped:

- `PGOLF-601` / `#188` retired the explicit family-level public CUDA blocker
  list in the canonical coverage report
- `PGOLF-604` / `#194` would have turned the bring-up seam into a real
  Psionic-native single-H100 trainer path, but that issue closed not planned
  when the lane stopped
- `PGOLF-605` / `#195` preserves the machine-readable single-H100 parity or
  refusal evidence
- `PGOLF-602` / `#189` would have been the later real `8xH100` evidence
  capture step, but that issue also closed not planned when the lane stopped
