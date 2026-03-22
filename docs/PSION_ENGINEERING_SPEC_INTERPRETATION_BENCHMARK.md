# Psion Engineering Spec-Interpretation Benchmark

> Status: canonical `PSION-23` / `#379` engineering-interpretation benchmark
> contract, written 2026-03-22 after landing the first typed engineering
> spec-interpretation package and its direct acceptance binding for `Psion`.

This document freezes the first dedicated engineering spec-interpretation
benchmark package for the `Psion` learned-model lane.

It is distinct from normative reading. The package is meant to test what
follows for implementations, where ambiguity matters, what remains
unspecified, and where portability risk appears, without claiming the source
text said more than it did.

## Canonical Artifacts

- `fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json` contains the
  canonical package row `psion_engineering_spec_benchmark_v1`.
- `fixtures/psion/benchmarks/psion_benchmark_receipt_set_v1.json` contains the
  canonical package receipt for that benchmark package.
- `fixtures/psion/benchmarks/psion_benchmark_label_generation_receipt_set_v1.json`
  contains the canonical label-generation receipt for that package.
- `fixtures/psion/acceptance/psion_acceptance_matrix_v1.json` now binds the
  engineering-interpretation requirements directly to
  `psion_engineering_spec_benchmark_v1` instead of leaving engineering
  interpretation as an ungated shared-package row.
- `crates/psionic-train/src/psion_benchmark_packages.rs` owns the typed
  engineering item payloads inside the shared package contract.
- `crates/psionic-train/src/psion_acceptance_matrix.rs` owns the direct
  benchmark-artifact binding used by the acceptance matrix.

## Package Identity

The first dedicated package is:

- package id: `psion_engineering_spec_benchmark_v1`
- package family: `engineering_spec_interpretation`
- package digest:
  `2223c33a7301921ee440a29936d734428ebd6c4a507a12fe4e60c76afad5445a`

## Typed Coverage

The canonical package now covers four engineering probe kinds explicitly:

- implementation implication
- ambiguity risk
- unspecified region
- portability consequence

Each engineering item now preserves:

- `normative_source_ref`
- `required_section_anchor`
- `probe_kind`
- `implementation_target`
- `expected_consequence`
- `normative_boundary_required`
- `explicit_uncertainty_required`
- `unsupported_certainty_forbidden`

This keeps the package focused on bounded engineering inference instead of
accepting confident implementation advice that hides where the spec is silent
or ambiguous.

## Labels And Receipts

The engineering package stays deliberately hybrid on the label side:

- the package receipt stays on the shared `PsionBenchmarkPackageReceipt`
  contract
- rubric-backed items score implementation implications, ambiguity, and
  portability reasoning without collapsing them into one exact string
- the exact-label item pins deterministic recognition of an unspecified region
  where the answer must admit that the implementation policy is outside the
  normative text
- the acceptance-ready evidence now lands under the dedicated
  `engineering_spec_interpretation` family

## Contamination Attachment

The package carries contamination attachment through the committed
contamination-input bundle:

- benchmark-visible source ids remain explicit
- held-out and training-excluded review inputs remain explicit
- the near-duplicate review reference remains explicit
- the package still only counts as green at promotion time when the phase gate
  carries a clean contamination review

## Acceptance Binding

`Psion` acceptance-matrix `v1` now binds the engineering-interpretation gate
directly to the concrete benchmark package artifact above.

The package is required from `broader_pretraining` onward:

- `broader_pretraining`: minimum `7500` bps
- `sft_promotion`: minimum `7900` bps
- `internal_serving`: minimum `8100` bps
- `trusted_cluster_scale_up`: minimum `8200` bps

This keeps engineering interpretation separate from normative reading in two
ways:

- the normative package remains the gate for what the text explicitly says
- the engineering package separately gates what follows for implementations
  once ambiguity, unspecified regions, and portability risks are handled
  honestly
