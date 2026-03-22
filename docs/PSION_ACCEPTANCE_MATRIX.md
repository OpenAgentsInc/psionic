# Psion Acceptance Matrix

> Status: canonical `PSION-5` / `#361` acceptance contract, written
> 2026-03-22 after landing the first `Psion` governance tranche through
> `docs/PSION_CORPUS_ADMISSION.md`,
> `docs/PSION_SOURCE_LIFECYCLE.md`, and
> `docs/PSION_BENCHMARK_ISOLATION.md`.

This document freezes `Psion` acceptance-matrix `v1` and the corresponding
phase-promotion decision contract.

It is not a claim that the full `Psion` lane is already green.

It is the machine-readable contract that says what evidence must exist before
the lane can move from pilot to broader pretraining, SFT promotion, internal
serving, or trusted-cluster scale-up.

## Canonical Artifacts

- `docs/PSION_ACCEPTANCE_MATRIX.md` is the canonical human-readable contract.
- `fixtures/psion/acceptance/psion_acceptance_matrix_v1.json` is the frozen
  matrix artifact.
- `fixtures/psion/acceptance/psion_promotion_decision_receipt_v1.json` is the
  first canonical promotion-decision receipt example.
- `crates/psionic-train/src/psion_acceptance_matrix.rs` owns the typed matrix,
  promotion receipt, and recording ledger.

## v1 Scope

`Psion` acceptance-matrix `v1` requires explicit gates for:

- `pilot`
- `broader_pretraining`
- `sft_promotion`
- `internal_serving`
- `trusted_cluster_scale_up`

Each gate carries:

- required benchmark families and threshold bands
- replay and checkpoint-recovery receipts
- contamination review requirements bound to
  `psion.benchmark_isolation.v1`
- route-calibration bands across direct answer, exact-executor handoff, and
  refusal
- refusal-calibration bands with explicit over-refusal ceilings

The pilot success criteria are intentionally frozen in one place:

- architecture reasoning must clear a named pass-rate floor
- held-out technical reasoning must improve over the seed baseline
- unsupported requests must be refused at the declared minimum band
- later scale-up remains blocked without a clean contamination review

## Promotion Contract

`Psion` phase promotion is recorded through the training-owned
`PsionPhasePromotionLedger`.

That ledger rejects a record unless the decision is attached to:

- the named acceptance-matrix id and version
- every benchmark receipt required by that phase
- replay evidence
- checkpoint-recovery evidence
- contamination review evidence
- route-calibration evidence
- refusal-calibration evidence

It also rejects inconsistent decisions:

- `promoted` is invalid if any gate is still red
- `held` is invalid if it does not list the exact gate-failure reasons

## Why This Matters

This closes the first promotion-governance step for the learned-model lane:

- one named `Psion` acceptance matrix now exists
- one typed phase-promotion decision contract now exists
- no phase decision can be recorded through the canonical ledger without the
  attached evidence set

Later issues still own the actual benchmark packages, pilot runs, larger
training runs, and trusted-cluster receipts. This doc only freezes the gate
that those later artifacts have to satisfy.
