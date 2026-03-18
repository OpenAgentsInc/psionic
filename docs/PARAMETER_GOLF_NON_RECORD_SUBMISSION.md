# Psionic Parameter Golf Non-Record Submission Package

> Status: canonical `PGOLF-401` / `#172` non-record submission contract,
> updated 2026-03-18 after landing the first typed package builder in
> `crates/psionic-train/src/parameter_golf_submission.rs`.

This document records the first honest Psionic packaging answer for Parameter
Golf.

It is intentionally a non-record answer.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfNonRecordSubmissionConfig`
- `ParameterGolfSubmissionAccountingReceipt`
- `ParameterGolfNonRecordSubmissionManifest`
- `ParameterGolfNonRecordSubmissionPackage`
- `ParameterGolfNonRecordSubmissionBundle`
- `build_parameter_golf_non_record_submission_bundle(...)`
- `write_parameter_golf_non_record_submission_bundle(...)`

The generated package now includes:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- the counted int8+zlib model artifact
- preserved benchmark-package, score-report, benchmark-receipt, accounting, and
  run-bundle JSON artifacts

## Folder Contract

The package root is a challenge-style
`records/track_non_record_16mb/<submission_id>` folder.

Inside that root, Psionic now emits:

- top-level human-facing files that match the public submission shape
- the preserved benchmark artifacts under the original run-relative Psionic
  paths
- one machine-readable accounting receipt that keeps counted bytes explicit

This is now a real record-folder output contract, not only prose in the
roadmap.

## Wrapper Posture

The shipped `train_gpt.py` is deliberately narrow.

It is a Python-stdlib review wrapper that:

- loads `submission.json`
- loads the preserved benchmark receipt
- loads the preserved accounting receipt
- verifies that the final score and counted bytes agree
- prints the final metric lines from the packaged run

That means the wrapper is runnable and self-contained, but it does **not**
pretend that Psionic has already closed the record-track counted-runtime story.

## Artifact Accounting

The package now keeps the public counted components explicit:

- `entrypoint_code_bytes`
- `compressed_model_bytes`
- `shipped_runtime_code_bytes`
- `shipped_wrapper_code_bytes`
- `required_build_dependency_bytes`

For the first landed package:

- the counted entrypoint is the generated top-level `train_gpt.py`
- the counted model is the int8+zlib artifact from the local-reference lane
- shipped runtime, extra wrapper, and build-dependency bytes are `0` because
  the package does not ship a Rust runtime payload or build tree

This is the intended honesty bar for the non-record lane: do not hide runtime
bytes, but also do not invent runtime bytes that are not actually shipped.

## Current Honest Boundary

This issue closes the non-record packaging lane only.

It does not claim:

- record-track runtime closure
- a defended counted-runtime story for an `8xH100` submission
- a green record-track readiness category

Those remain explicit follow-on work.
