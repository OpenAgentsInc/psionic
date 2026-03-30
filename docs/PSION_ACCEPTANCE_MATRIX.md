# Psion Acceptance Matrix

> Status: canonical `PSION-5` / `#361` acceptance contract, written
> 2026-03-22 after landing the first `Psion` governance tranche through
> `docs/PSION_CORPUS_ADMISSION.md`,
> `docs/PSION_SOURCE_LIFECYCLE.md`, and
> `docs/PSION_BENCHMARK_ISOLATION.md`.

This document freezes `Psion` acceptance-matrix `v1` and the corresponding
phase-promotion decision contract.

This document is the generic compact-decoder learned-lane acceptance contract.

Executor-capable `Psion` / `Tassadar` promotion and replacement work uses
`docs/PSION_EXECUTOR_ACCEPTANCE_PROFILE.md` instead of flattening the executor
lane into this generic matrix.

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

`PSION-21` tightens the architecture gate further: the architecture reasoning
requirements now bind directly to the concrete benchmark package
`psion_architecture_reasoning_benchmark_v1` instead of only naming the family.
Pilot and later scale-up decisions therefore cannot satisfy the architecture
gate with an arbitrary receipt from another package.

`PSION-22` applies the same pattern to normative source-grounded reading:
`psion_normative_spec_benchmark_v1` now has its own named acceptance family
and direct package binding, so normative reading no longer rides on the older
combined spec/manual benchmark family or on engineering interpretation receipts.

`PSION-23` adds the parallel engineering interpretation gate:
`psion_engineering_spec_benchmark_v1` is now bound directly into broader
pretraining, SFT promotion, internal serving, and trusted-cluster scale-up.
That keeps implementation implications and portability reasoning explicit
without pretending the pilot already proves the full engineering lane.

`PSION-24` adds a dedicated memorization-versus-reasoning gate:
`psion_memorization_reasoning_benchmark_v1` now binds directly into
`sft_promotion` and later phase gates, so recombination failures on altered
constraints, paraphrases, historical transfer, and spec-adjacent edge cases no
longer hide inside the generic held-out technical reasoning family.

`PSION-25` tightens route selection the same way:
`psion_route_benchmark_v1` now binds directly into the later route-selection
gates, and the route lane now emits a dedicated route-class receipt that keeps
answer-in-language, answer-with-uncertainty, request-for-structure, and
exact-delegation outcomes separate from refusal work.

`PSION-26` applies the same direct-binding rule to refusal:
`psion_unsupported_request_refusal_benchmark_v1` now binds directly into every
phase that carries the unsupported-request refusal family, and the refusal lane
now emits a dedicated refusal-calibration receipt that keeps capability region,
reason-code match, supported-control over-refusal, and regression visible
instead of collapsing the whole surface into one refusal percentage.

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

`PSION-30` now explicitly reuses this same acceptance contract for contributed
outputs through `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`. Local window
promotion in the bounded decentralized lane is therefore not treated as a
replacement for `Psion` phase promotion.
