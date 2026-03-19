# Psionic Parameter Golf Record-Folder Compatibility

> Status: canonical `PGOLF-501` / `#184` challenge-repo folder-compatibility
> contract, written 2026-03-18 after landing
> `crates/psionic-train/src/parameter_golf_record_folder_compatibility.rs`,
> `fixtures/parameter_golf/reports/parameter_golf_record_folder_compatibility.json`,
> and `scripts/check-parameter-golf-record-folder-compatibility.sh`.

This document freezes the first explicit answer to a narrow but important
question:

> can a Psionic-exported Parameter Golf folder be checked as a drop-in
> `parameter-golf/records/...` submission folder instead of only as a
> repo-local package shape?

It is intentionally a folder-compatibility gate, not a record-track claim.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfRecordFolderCompatibilityReport`
- `build_parameter_golf_record_folder_compatibility_report()`
- `write_parameter_golf_record_folder_compatibility_report(...)`

The committed machine-readable report now lives at:

- `fixtures/parameter_golf/reports/parameter_golf_record_folder_compatibility.json`

The canonical verifier command now lives at:

- `scripts/check-parameter-golf-record-folder-compatibility.sh`

## Public Contract Frozen Here

As of 2026-03-18, the local `~/code/parameter-golf` repo says:

- submissions land under either `records/track_10min_16mb` or
  `records/track_non_record_16mb`
- each submission folder must include `README.md`, `submission.json`,
  `train.log`, and `train_gpt.py`
- the folder may ship extra dependencies if they live inside the folder and
  the entrypoint still compiles or runs in place
- broken entrypoints are refused

The compatibility report keeps that contract machine-readable and binds it to
the current Psionic non-record export surface.

The current exported folder now satisfies that contract with a root-local
`train_gpt.py` launcher plus a shipped Psionic runtime payload under
`runtime/`.

## Canonical Gate

Validate the committed report only:

```bash
scripts/check-parameter-golf-record-folder-compatibility.sh
```

Dry-run one exported folder against the local public repo clone:

```bash
scripts/check-parameter-golf-record-folder-compatibility.sh \
  --parameter-golf-root ~/code/parameter-golf \
  --submission-dir /tmp/records/track_non_record_16mb/<submission_id>
```

Write a machine-readable verifier result:

```bash
scripts/check-parameter-golf-record-folder-compatibility.sh \
  --parameter-golf-root ~/code/parameter-golf \
  --submission-dir /tmp/records/track_non_record_16mb/<submission_id> \
  --report /tmp/parameter_golf_record_folder_verification.json
```

This verifier is now the canonical gate for challenge-repo folder
compatibility.

## Current Honest Boundary

This issue closes only the folder-compatibility question.

It does not claim:

- record-track runtime closure
- a defended counted-runtime or build-dependency story for a stronger future
  record-track payload
- reproducible `8xH100` record-track execution from the exported folder itself

So the strongest honest claim posture remains `non_record_submission`.
