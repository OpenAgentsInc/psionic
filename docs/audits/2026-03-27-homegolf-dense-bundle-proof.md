# HOMEGOLF Dense Bundle Proof

> Status: retained 2026-03-27 proof for `HOMEGOLF-2`

This audit records the first HOMEGOLF train-to-infer closure proof.

After `HOMEGOLF-8`, this is no longer the canonical runnable contest-lane
surface. The strict runnable lane is now:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_strict_challenge_lane.rs`
- `fixtures/parameter_golf/reports/parameter_golf_homegolf_strict_challenge_lane.json`

This dense bundle proof remains the secondary exact-family train-to-infer proof.

## What Ran

Runner:

- `crates/psionic-serve/examples/parameter_golf_homegolf_dense_bundle_proof.rs`

Retained proof report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json`

Source dense baseline surface:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_baseline_surface.json`

## What It Proves

- the emitted model family is still the exact public `SP1024` `9x512` baseline
- Psionic can emit a real promoted runtime bundle from an exact-family bounded
  HOMEGOLF-compatible run
- that bundle loads for direct local inference
- that same bundle loads through `psionic-serve`
- direct and served generation match on the retained proof prompt

## Retained Result

- `run_id=parameter-golf-homegolf-dense-bundle-proof`
- `profile_id=psion_small_decoder_pgolf_core_v0`
- `descriptor_digest=8a111f908acf02174554a75e83e13092852ee7caa534f65c43be897bd4c606ee`
- `tokenizer_digest=49b44264442058c20b2b95a947f3aac60e8729fd57d63db8b8754de8edb98a6d`
- `model_artifact_bytes=68248296`
- `final_validation_mean_loss=8.60598874092102`
- `final_validation_bits_per_byte=9.93265382277841`
- prompt tokens: `[1, 2, 3, 4]`
- direct generated tokens: `[7, 2, 7, 2]`
- served generated tokens: `[7, 2, 7, 2]`
- direct/served parity: `true`

## Honest Boundary

This closes the train-to-infer mechanics for HOMEGOLF at the exact-family level.

It does not prove:

- that the retained single-H100 dense baseline source report already shipped the
  final model bytes needed to rebuild a bundle directly from that exact run
- that the public FineWeb/SP1024 challenge scorepath is closed locally
- that mixed-device `10` minute HOMEGOLF execution is already complete

So the repo truth now is:

- exact dense baseline surface: yes
- exact-family promoted bundle closure: yes
- direct and served inference parity on that emitted bundle: yes
- clustered HOMEGOLF score run: still later work
