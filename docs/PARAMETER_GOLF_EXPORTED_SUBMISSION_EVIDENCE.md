# Psionic Parameter Golf Exported Submission Evidence

> Status: canonical `PGOLF-603` / `#190` exported-folder evidence and
> replay-verification contract, updated 2026-03-24 after landing
> `crates/psionic-train/src/parameter_golf_submission_pr.rs`.

This document records the challenge-facing evidence Psionic now preserves from
the exported submission folder itself.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfSubmissionRunEvidenceReport`
- `ParameterGolfRecordFolderReplayVerificationReport`
- `build_parameter_golf_submission_run_evidence_report(...)`
- `build_parameter_golf_record_folder_replay_verification_report(...)`
- `write_parameter_golf_submission_run_evidence_report(...)`
- `write_parameter_golf_record_folder_replay_verification_report(...)`

The repo now also ships:

- `scripts/check-parameter-golf-record-folder-replay.sh`
- `fixtures/parameter_golf/reports/parameter_golf_submission_run_evidence.json`
- `fixtures/parameter_golf/reports/parameter_golf_record_folder_replay_verification.json`

## Exported-Folder Evidence

The run-evidence report now binds one exported folder to:

- the exact shipped `train_gpt.py`, runtime payload, runtime manifest, train
  log, model artifact, accounting receipt, benchmark receipt, run bundle, and
  runtime receipt digests
- one real folder-local entrypoint execution
- the preserved bounded wallclock, memory, and artifact-size receipts
- one measured-or-refused `8xH100` challenge receipt for the same folder

The exported folder now also carries the real single-H100 trainer payload, the
real-execution contract, and the immutable PGOLF input-package descriptor. The
current replay verifier still exercises only the default bounded local-
reference mode; later remote evidence binds those extra shipped bytes to the
same exported-folder surface rather than introducing a second package shape.

That closes the earlier gap where these facts existed only on the internal
benchmark path instead of on the exported folder surface itself.

The canonical exported folder now also freezes the bounded local-reference
wallclock receipt to one committed reference measurement so the generated
non-record evidence bundle stays deterministic across reruns. That stability is
for replay and accounting review only; it is not a challenge-speed claim.

## Replay Verification

The replay verifier now checks the exported folder directly:

- `submission.json` versus the final `train.log` metrics
- `submission.json` versus the preserved benchmark receipt metrics
- `submission.json` versus the runtime receipt metrics
- `submission.json.wallclock_seconds` versus the preserved benchmark receipt
- counted-byte facts versus the shipped accounting receipt and actual file sizes

Run the verifier from the folder root:

```bash
scripts/check-parameter-golf-record-folder-replay.sh
```

Target another folder and preserve the report:

```bash
scripts/check-parameter-golf-record-folder-replay.sh \
  --submission-dir /tmp/records/track_non_record_16mb/<submission_id> \
  --report /tmp/parameter_golf_record_folder_replay_verification.json
```

Generate the exported-folder evidence report directly:

```bash
cargo run -p psionic-train --example parameter_golf_submission_run_evidence \
  /tmp/records/track_non_record_16mb/<submission_id> \
  /tmp/parameter_golf_submission_run_evidence.json
```

## Current Honest Boundary

These reports close exported-folder evidence and replay verification only.

They do not claim:

- that real exported-folder `8xH100` evidence exists
- that the current committed evidence is a true `8xH100` success
- that the remaining CUDA baseline blockers are retired
- that the current non-record folder is record-ready

The current committed challenge receipt is still a refusal on the local
single-`RTX 4080` review host. That is intentional: the point of this layer is
to make the folder-facing evidence and refusal posture explicit rather than
implied. The repo never reached true exported-folder `8xH100` evidence before
the lane stopped.
