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

The current lane claim posture is `research`.

That means:

- the acceptance contract is now canonical and runnable
- claim language is now frozen by `docs/PARAMETER_GOLF_ACCOUNTING.md`
- later implementation work must close named categories instead of using
  free-form benchmark claims
- most technical categories are still honestly red today

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
| `single-device-trainer-parity` | `partial` | A Psionic-owned baseline decoder can train, validate, export, reload, and re-evaluate on a single device while matching the public baseline architecture and optimizer behavior. | `psionic-models` now ships the compact Parameter Golf decoder family with stable tensor naming, baseline parameter accounting, and a frozen `train_gpt.py` parity fixture at the public `9x512` shape, and `psionic-train` now ships the challenge optimizer split, baseline schedule helpers, exact Muon reference math, and a frozen `train_gpt.py` optimizer fixture, but the single-device trainer and export or reload lane above them are still missing. | `PGOLF-201` / `#166`, `PGOLF-202` / `#167`, `PGOLF-203` / `#168` | A promising architecture sketch or one-off local run is not enough. |
| `distributed-throughput-closure` | `planned` | Psionic has one explicit `8xH100` execution path with topology, timing, memory, and artifact receipts that support the declared challenge bar. | No benchmark package, no distributed `8xH100` trainer lane, and no widened CUDA training surface are landed for Parameter Golf yet. | `PGOLF-301` / `#169`, `PGOLF-302` / `#170`, `PGOLF-303` / `#171` | Do not infer `8xH100` closure from local or reference-emulated distributed helpers. |
| `packaging-readiness` | `partial` | Psionic can emit one self-contained submission folder with explicit artifact-byte accounting, runnable entrypoint shape, record metadata, and preserved receipts. | Claim language and accounting posture are now explicit in `docs/PARAMETER_GOLF_ACCOUNTING.md`, but no runnable submission wrapper, `submission.json` generator, README generator, or record-folder output contract is landed yet. | `PGOLF-002` / `#161`, `PGOLF-401` / `#172` | This category is intentionally ahead of technical closure only at the governance level; the actual package is still missing. |
| `record-track-readiness` | `partial_outside_psionic` | Psionic can defend a record-track submission under the published challenge rules, including counted code, self-contained evaluation, and reproducible `8xH100` execution. | The public challenge language still centers `train_gpt.py` and counted script bytes, while Psionic has not yet landed either the technical lane or the final counted-runtime story. | `PGOLF-002` / `#161`, `PGOLF-302` / `#170`, `PGOLF-401` / `#172`, `PGOLF-403` / `#174` | Treat this as blocked by both in-repo implementation work and the need for an explicit public accounting interpretation. |

## Why This Matters

This issue closes the second governance requirement in
`docs/ROADMAP_PARAMETERGOLF.md`:

- one Parameter Golf acceptance matrix exists
- one machine-readable acceptance report exists
- one repo-owned checker exists

Epic 0 governance is now complete because the roadmap, accounting contract, and
acceptance matrix all exist as runnable repo truth.
