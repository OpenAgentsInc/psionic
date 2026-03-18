# Psionic Parameter Golf Accounting And Claim Language

> Status: canonical `PGOLF-002` / `#161` accounting and claim-language
> contract, written 2026-03-18 after reviewing
> `~/code/parameter-golf/README.md`,
> `~/code/parameter-golf/PSIONIC_PARAMETER_GOLF_SPEC.md`, and
> `docs/ROADMAP_PARAMETERGOLF.md`.

This document freezes how Psionic talks about Parameter Golf claims before the
lane has full oracle, training, throughput, and packaging closure.

It exists because the public challenge language is already specific:

- the artifact cap is decimal `16,000,000` bytes
- the submission artifact is defined as code bytes plus compressed model bytes
- counted code should live in `train_gpt.py`
- evaluation must be self-contained, reproducible, and offline

That means Psionic cannot honestly hide a Rust runtime, a vendored build tree,
or a wrapper dependency stack behind a tiny launcher and still call the result
record-ready.

## Canonical Posture

The canonical current Psionic claim posture is:

- `non_record_submission`

That is the honest posture on 2026-03-18 because:

- exact challenge-oracle parity is landed
- the repo now has a real non-record submission-folder output contract with
  `README.md`, `submission.json`, `train.log`, a `train_gpt.py`-shaped wrapper,
  and explicit counted-byte accounting
- the baseline single-device trainer is still only a bounded reference path
- the `8xH100` distributed receipt lane is landed, but distributed-throughput
  closure is still not green
- the counted-runtime story for a record-track Psionic-owned submission is
  still not defensible

## Claim Vocabulary

Use only the following claim labels for the Parameter Golf lane.

| Claim label | Meaning | Minimum closure required | Forbidden shorthand |
| --- | --- | --- | --- |
| `research` | Psionic is experimenting on the lane, but the result is not yet a submission claim. | none beyond explicit artifact and benchmark honesty | do not call it `submission-ready`, `leaderboard-ready`, `record-track`, or `challenge-compliant` |
| `non_record_submission` | Psionic can produce a self-contained submission package that is honest about artifact accounting, but it is not a record-track claim. This may be because the run exceeds the `10` minute cap or because it is intentionally filed as a non-record entry. | challenge-oracle parity plus packaging readiness | do not call it `record-ready` or imply the `8xH100` record bar is closed |
| `record_candidate_blocked_on_accounting` | Psionic may have the model, training, and throughput closure needed for a serious record attempt, but the public accounting or wrapper story is still not defensible. | challenge-oracle parity, single-device trainer parity, and distributed throughput closure; packaging may still be blocked on counted-code or entrypoint rules | do not call it `leaderboard-ready`, `challenge-compliant`, or `record-track ready` |
| `record_ready` | Psionic can defend a record-track submission under the published challenge rules, including counted code, self-contained evaluation, and reproducible record-folder output. | all acceptance categories green | do not use unless the acceptance matrix and packaging contract are both green |

If a future doc, issue, benchmark note, or README uses stronger language than
the current claim label allows, that text is wrong and must be corrected before
the lane can claim progress.

## Accounting Rules

These rules are now canonical for Psionic-owned Parameter Golf work.

### 1. Count all shipped evaluation code

If evaluation requires code to be present in the submission folder, those bytes
count.

For Psionic this means:

- a Rust source tree copied into the submission folder counts
- a prebuilt Rust binary copied into the submission folder counts
- generated support code or glue code copied into the submission folder counts
- a Python wrapper plus a Rust binary both count when both are required

### 2. A thin wrapper does not make the runtime free

If `train_gpt.py` is only a launcher that shells out to a large Rust binary or
to a vendored multi-crate workspace, Psionic must count the shipped runtime
bytes and helper code bytes needed for evaluation.

The wrapper is acceptable only when:

- challenge rules still require a `train_gpt.py`-shaped entrypoint
- the wrapper is honestly counted when it ships
- every additional shipped dependency required by the wrapper is honestly
  counted too

### 3. Build-time dependencies are not exempt by default

If the submission requires a compiler, vendored crates, generated code, or
other build-time dependencies to produce the runnable artifact during
evaluation, Psionic must assume those bytes matter unless the public challenge
rules explicitly say otherwise.

Until that exemption exists in the public rules, the default Psionic posture
is:

- do not assume Cargo dependencies are free
- do not assume a system Rust toolchain is free
- do not assume code outside the record folder is free

### 4. Offline and self-contained evaluation is mandatory

The submission may not depend on:

- network access
- external model downloads
- training-dataset access during evaluation
- tokenizer or support artifacts that are fetched from outside the submission

If an artifact is required during evaluation, Psionic must either ship it
inside the counted submission surface or refuse the stronger claim.

### 5. Record-track language stays blocked until the code-accounting story is explicit

Psionic may only claim `record_ready` when it has a defensible answer for all
of the following:

- what exact files are shipped in the record folder
- which of those files count as code bytes
- which of those files count as compressed model bytes
- how the shipped entrypoint executes without hidden external code
- how the result stays self-contained under the public challenge rules

Absent that answer, the strongest allowed language is either
`non_record_submission`, `research`, or
`record_candidate_blocked_on_accounting`, depending on the acceptance matrix.

## Claim-To-Matrix Mapping

The canonical acceptance gate for these claims lives in
`docs/PARAMETER_GOLF_ACCEPTANCE_MATRIX.md` and
`fixtures/parameter_golf/reports/parameter_golf_acceptance_report.json`.

The required mapping is:

- `research`
  - default posture when any foundational category is still red
- `non_record_submission`
  - requires `challenge-oracle-parity` and `packaging-readiness`
- `record_candidate_blocked_on_accounting`
  - requires `challenge-oracle-parity`, `single-device-trainer-parity`, and
    `distributed-throughput-closure`, while `record-track-readiness` remains
    blocked
- `record_ready`
  - requires every category in the acceptance matrix to be green

This split is deliberate. A strong training result without an honest packaging
story is not a record-track claim, and a polished submission wrapper without
oracle parity is not a valid challenge claim either.

## What Changes This Doc

This document should change only when one of these things changes:

- the public Parameter Golf rules change
- Psionic lands a new counted-code or entrypoint story
- the acceptance matrix turns a stronger claim class green

If a future issue wants to broaden claim language without updating this
document, that issue is blocked by definition.
