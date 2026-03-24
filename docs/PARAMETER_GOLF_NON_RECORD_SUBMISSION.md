# Psionic Parameter Golf Non-Record Submission Package

> Status: canonical `PGOLF-401` / `#172` plus `PGOLF-502` / `#185`
> non-record submission contract, updated 2026-03-24 after landing the real
> shipped submission runtime path in
> `crates/psionic-train/src/parameter_golf_submission.rs` and
> `crates/psionic-train/src/parameter_golf_submission_runtime.rs`.

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
- `runtime/parameter_golf_submission_runtime`
- `runtime/parameter_golf_single_h100_train`
- `runtime/parameter_golf_submission_runtime.json`
- `runtime/parameter_golf_real_execution_contract.json`
- `runtime/parameter_golf_local_reference_fixture.json`
- `runtime/parameter_golf_google_input_package_descriptor_v1.json`
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

## Challenge-Repo Compatibility Gate

The canonical folder-compatibility gate now lives in:

- `docs/PARAMETER_GOLF_RECORD_FOLDER_COMPATIBILITY.md`
- `fixtures/parameter_golf/reports/parameter_golf_record_folder_compatibility.json`
- `scripts/check-parameter-golf-record-folder-compatibility.sh`

Run the verifier against one exported folder and the local public repo clone:

```bash
scripts/check-parameter-golf-record-folder-compatibility.sh \
  --parameter-golf-root ~/code/parameter-golf \
  --submission-dir /tmp/records/track_non_record_16mb/<submission_id>
```

Export the canonical Psionic folder directly into the live challenge repo:

```bash
cargo run -p psionic-train --example parameter_golf_non_record_submission_bundle \
  ~/code/parameter-golf/records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2
```

That export path is the first honest maintainer-facing Psionic submission
surface for the public non-record lane.

The canonical export path now also freezes the bounded local-reference
wallclock receipt to one committed reference measurement so the exported-folder
digests stay stable across reruns. That keeps the non-record review surface
deterministic without pretending it is challenge-speed timing evidence.

The exported folder now also has explicit follow-on evidence and PR helpers:

- `docs/PARAMETER_GOLF_EXPORTED_SUBMISSION_EVIDENCE.md`
- `docs/PARAMETER_GOLF_PR_SUBMISSION_FLOW.md`
- `scripts/check-parameter-golf-record-folder-replay.sh`
- `cargo run -p psionic-train --example parameter_golf_final_pr_bundle`

That gate is intentionally narrower than record-track readiness. It proves the
current non-record folder matches the live `records/...` contract and dry-runs
inside a challenge-shaped repo root. It does **not** upgrade the current
bounded local-reference runtime replay into a record-track claim.

## Runtime Posture

The shipped `train_gpt.py` is now deliberately small rather than review-only.

It is a Python-stdlib launcher that:

- defaults to the shipped `runtime/parameter_golf_submission_runtime` payload
- passes the shipped runtime manifest for the bounded local-reference replay path
- can also dispatch to the shipped `runtime/parameter_golf_single_h100_train`
  payload when the explicit execution-mode environment contract is supplied
- keeps the folder self-contained and root-local

The default shipped runtime payload then:

- restores the included int8+zlib model artifact
- re-runs the bounded local-reference validation path on the shipped fixture
- checks consistency against `submission.json` and the shipped accounting
  receipt
- writes `parameter_golf_submission_runtime_receipt.json` inside the folder

The same exported folder now also ships:

- one machine-readable real-execution contract
- the immutable PGOLF input-package descriptor used by the later Google and
  RunPod operator lanes
- one prebuilt single-H100 trainer payload that `train_gpt.py` can invoke in
  `single_h100_train` mode when the dataset and tokenizer environment contract
  is provided

That means the package now owns both:

- a bounded local-reference replay path for challenge-clone dry runs
- a real exported-folder single-H100 trainer entry surface for later remote
  evidence

It still does **not** pretend that Psionic has already closed the record-track
counted-runtime story.

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
- the counted runtime is the shipped prebuilt pair
  `runtime/parameter_golf_submission_runtime` plus
  `runtime/parameter_golf_single_h100_train`
- extra wrapper bytes are `0` because no additional launcher layer is shipped
- build-dependency bytes are `0` because the folder ships a prebuilt payload
  and requires no in-folder build tree

This is the intended honesty bar for the non-record lane: do not hide runtime
bytes, and do not invent runtime bytes that are not actually shipped.

## Current Honest Boundary

This issue closes the non-record packaging lane only.

It does not claim:

- record-track runtime closure
- a defended counted-runtime story for an `8xH100` submission
- a green record-track readiness category

The exported folder can now invoke the real single-H100 trainer surface, but
record-track accounting and distributed `8xH100` evidence remain explicit
follow-on work.
