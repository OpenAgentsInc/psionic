# HOMEGOLF Live Dense Run Surface Audit

Date: March 27, 2026

## Summary

This audit records the upgraded HOMEGOLF surface that replaces the earlier
open-adapter-plus-bounded-bundle surrogate with dense retained sources only.

Retained machine-readable report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json`

Generator and checker:

- `crates/psionic-train/src/parameter_golf_homegolf_clustered.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_clustered_run_surface.rs`
- `scripts/check-parameter-golf-homegolf-clustered-run-surface.sh`

## What Landed

The canonical HOMEGOLF surface now binds together two stronger retained sources:

- one real same-job dense mixed-device runtime proof:
  `fixtures/training/first_same_job_mixed_backend_dense_run_v1.json`
- one real exact dense challenge export report:
  `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_training_report.json`

That means the HOMEGOLF track no longer depends on the older composition of:

- open-adapter home Tailnet receipts
- the bounded promoted HOMEGOLF bundle proof

for its main score surface.

## Retained Values

The upgraded surface keeps:

- participating hardware classes:
  - `local_apple_silicon_metal`
  - `optional_h100_node`
- retained dense wallclock:
  - `observed_cluster_wallclock_ms=3611`
  - `wallclock_cap_seconds=600`
- dense participant receipts:
  - `runpod-cuda-submesh`
    - `estimated_steps_per_second=1.6339869281045751`
    - `estimated_samples_per_second=963764.705882353`
    - `contribution_share=0.8888888888888888`
  - `local-mlx-rank`
    - `estimated_steps_per_second=1.4270424545130218`
    - `estimated_samples_per_second=841703.8886906885`
    - `contribution_share=0.1111111111111111`
- exact dense score/export bindings:
  - `merged_bundle_descriptor_digest=af7583b983cd2016d1c4aa7b8e185557048539f3be50db26873247e4d2bc9981`
  - `merged_bundle_tokenizer_digest=4f5e8adb109c66b4886963bc75a7befd73bda36d27fd7102df8e9e66503b0e2a`
  - `scored_model_artifact_ref=parameter-golf-single-h100-trainer/step-00001/final_model.int8.ptz`
  - `scored_model_artifact_digest=4657d793ae3e64796670b6768f433c48f788518725ba2854e913db205412b250`
  - `model_artifact_bytes=4732744`
  - `final_validation_mean_loss=10.64899006955549`
  - `final_validation_bits_per_byte=6.306931747817168`

## Why This Surface Is Better

The previous HOMEGOLF clustered surface was honest, but weak:

- the clustered execution truth came from an open-adapter home run
- the score and artifact bytes came from a separate bounded promoted bundle

This upgraded surface is still not perfect, but it is materially stronger:

- the execution truth is now dense and mixed-device
- the scored bytes now come from the real exact dense challenge export path
- the scored artifact is now under the contest cap

## What This Proves

The repo can now truthfully say:

- HOMEGOLF has one retained dense mixed-device runtime surface
- that surface preserves real MLX-plus-CUDA participant receipts
- that same surface now carries the exact dense challenge export bytes instead
  of the old promoted-bundle surrogate
- the canonical HOMEGOLF score/comparison/accounting path is now anchored to
  dense retained sources only

## What This Does Not Prove

This surface still does **not** prove:

- admitted home-RTX dense closure on the current home cluster
- one single retained run id that already binds the mixed-device dense runtime
  receipts and the final scored export bytes in one artifact family
- public-leaderboard equivalence

Those remaining gaps belong to later HOMEGOLF work, not to this audit.

## Verification

The landed surface was revalidated with:

```bash
cargo check -q -p psionic-train

cargo run -q -p psionic-train --bin parameter_golf_homegolf_clustered_run_surface -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json

./scripts/check-parameter-golf-homegolf-clustered-run-surface.sh
```
