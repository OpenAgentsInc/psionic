# Psionic Parameter Golf Acceptance Matrix

> Status: canonical `PGOLF-003` / `#162` acceptance contract, written
> 2026-03-18 after landing `docs/PARAMETER_GOLF_ACCOUNTING.md`,
> `fixtures/parameter_golf/reports/parameter_golf_acceptance_report.json`, and
> `scripts/check-parameter-golf-acceptance.sh`.

This document defines the category gates for the Psionic Parameter Golf lane.

It is not a claim that the lane is already green.

It is the contract that says what has to become true before stronger Parameter
Golf claims become honest.

## Canonical Runner

Run the matrix from the repo root:

```bash
scripts/check-parameter-golf-acceptance.sh
```

Write the machine-readable report:

```bash
scripts/check-parameter-golf-acceptance.sh \
  --report /tmp/psionic-parameter-golf-acceptance.json
```

Target one or more categories:

```bash
scripts/check-parameter-golf-acceptance.sh --only challenge-oracle-parity
scripts/check-parameter-golf-acceptance.sh --only packaging-readiness --only record-track-readiness
```

The report schema lives at
`docs/parameter_golf_acceptance_report.schema.json`.

## Current Posture

The initial machine-readable matrix posture is `tracking_only`.

The current lane claim posture is `non_record_submission`.

Active development on the lane stopped on 2026-03-19 by explicit user
direction. This matrix therefore remains historical claim truth for the landed
repo state rather than an active promotion target. See
`docs/PARAMETER_GOLF_AFTER_ACTION.md` for the stop record.

That means:

- the acceptance contract is now canonical and runnable
- claim language is now frozen by `docs/PARAMETER_GOLF_ACCOUNTING.md`
- the first honest non-record submission package is now landed
- record-track and distributed-throughput closure are still honestly partial
  today

## Claim Split

The acceptance matrix answers a narrower question than the roadmap:

> which concrete closure categories must be green before Psionic can use a
> stronger Parameter Golf claim label?

The required mapping is:

| Claim label | Required category posture |
| --- | --- |
| `research` | default while any foundational category remains red |
| `non_record_submission` | `challenge-oracle-parity` and `packaging-readiness` must be green |
| `record_candidate_blocked_on_accounting` | `challenge-oracle-parity`, `single-device-trainer-parity`, and `distributed-throughput-closure` must be green while `record-track-readiness` remains blocked |
| `record_ready` | every category in this matrix must be green |

If a later issue tries to close one claim with evidence from another category,
that issue is wrong by definition and should update this document first.

## Matrix

| Category | Current status | What a green category would mean | Current repo truth | Governing issues | Boundary note |
| --- | --- | --- | --- | --- | --- |
| `challenge-oracle-parity` | `implemented` | Psionic reproduces the current FineWeb shard contract, fixed validation split, exact `val_loss`, and exact `val_bpb` accounting against frozen challenge fixtures. | `psionic-data` now owns the shard ABI, validation token loader, SentencePiece byte-accounting LUT oracle, a committed parity fixture, and a frozen test that matches the current `train_gpt.py` and `train_gpt_mlx.py` reference paths for shard loading, validation slicing, `val_loss`, and `val_bpb`. | `PGOLF-101` / `#163`, `PGOLF-102` / `#164`, `PGOLF-103` / `#165` | Do not claim comparable Parameter Golf scores until this category is green. |
| `single-device-trainer-parity` | `implemented_early` | A Psionic-owned baseline decoder can train, validate, export, reload, and re-evaluate on a single device while matching the public baseline architecture and optimizer behavior. | `psionic-models` now ships the compact Parameter Golf decoder family with stable tensor naming, baseline parameter accounting, a frozen `train_gpt.py` parity fixture at the public `9x512` shape, and named-parameter export helpers, while `psionic-train` plus `psionic-eval` now ship a bounded local-reference trainer with explicit single-device challenge batch geometry, grad accumulation, checkpoint and restart state, raw safetensors export, int8+zlib roundtrip restore, and validation re-eval under Psionic ownership. This is real single-device lane ownership, but it is still a bounded CPU-reference path rather than measured challenge-scale throughput closure. | `PGOLF-201` / `#166`, `PGOLF-202` / `#167`, `PGOLF-203` / `#168` | Treat this as stronger than sketches or one-off runs, but not as `8xH100` closure or leaderboard-ready throughput. |
| `distributed-throughput-closure` | `partial` | Psionic has one explicit `8xH100` execution path with topology, timing, memory, and artifact receipts that support the declared challenge bar. | `psionic-train` plus `psionic-eval` now ship an exact `8xH100` DDP-style receipt lane aligned to the current `train_gpt.py` posture, including explicit CUDA-device admission gates, replicated topology, NCCL-style all-reduce communication stages, measured wallclock receipts, analytic distributed memory planning, and a machine-readable CUDA training coverage report digest plus blocker list carried on the same receipt seam. Full challenge-speed decoder-kernel and runtime widening are still honestly open. | `PGOLF-301` / `#169`, `PGOLF-302` / `#170`, `PGOLF-303` / `#171` | Treat this as explicit distributed execution truth with honest refusal posture and explicit CUDA blocker tracking, not as proof that the remaining CUDA kernel gaps are closed. |
| `packaging-readiness` | `implemented` | Psionic can emit one self-contained submission folder with explicit artifact-byte accounting, runnable entrypoint shape, record metadata, and preserved receipts. | `psionic-train` now ships a typed non-record submission package builder that emits `README.md`, `submission.json`, `train.log`, a root-local `train_gpt.py` launcher, a shipped Psionic runtime payload plus runtime manifest and fixture, the counted int8+zlib model artifact, preserved benchmark receipts, and a machine-readable counted-byte accounting receipt, plus a writer that materializes the folder contract to disk. | `PGOLF-002` / `#161`, `PGOLF-401` / `#172`, `PGOLF-502` / `#185` | This closes the first honest non-record package only; it does not imply record-track runtime closure or `8xH100` record readiness. |
| `record-track-readiness` | `partial_outside_psionic` | Psionic can defend a record-track submission under the published challenge rules, including counted code, self-contained evaluation, and reproducible `8xH100` execution. | Psionic now has a committed blocked record-track contract that binds the acceptance report, non-record package surface, exported-folder evidence and replay verification, typed promotion receipt posture, final PR-bundle generation, local challenge-clone dry run, research harness, and distributed benchmark reference together. The folder-local runtime and review surfaces are now explicit, but defended counted-runtime or build-dependency posture and reproducible challenge-speed `8xH100` execution remain explicit blockers. | `PGOLF-002` / `#161`, `PGOLF-302` / `#170`, `PGOLF-401` / `#172`, `PGOLF-403` / `#174`, `PGOLF-502` / `#185`, `PGOLF-603` / `#190`, `PGOLF-701` / `#191`, `PGOLF-702` / `#192`, `PGOLF-703` / `#193` | Treat this as stronger blocked truth, not as green record-track readiness. |

## Why This Matters

This issue closes the second governance requirement in
`docs/ROADMAP_PARAMETERGOLF.md`:

- one Parameter Golf acceptance matrix exists
- one machine-readable acceptance report exists
- one repo-owned checker exists

Epic 0 governance is now complete because the roadmap, accounting contract, and
acceptance matrix all exist as runnable repo truth.
