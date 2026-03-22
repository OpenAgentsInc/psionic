# Psion Memorization-Vs-Reasoning Probes

> Status: canonical `PSION-24` / `#380` memorization-versus-reasoning probe
> contract, written 2026-03-22 after landing the first typed recombination
> package and its direct acceptance binding for `Psion`.

This document freezes the first dedicated memorization-versus-reasoning probe
package for the `Psion` learned-model lane.

It is distinct from generic held-out technical reasoning. The package is meant
to test whether the model can recombine the learned ontology under changed
constraints, paraphrases, unfamiliar compositions, historical transfer, and
spec-adjacent edge cases, rather than reciting familiar stock passages.

## Canonical Artifacts

- `fixtures/psion/benchmarks/psion_benchmark_catalog_v1.json` contains the
  canonical package row `psion_memorization_reasoning_benchmark_v1`.
- `fixtures/psion/benchmarks/psion_benchmark_receipt_set_v1.json` contains the
  canonical package receipt for that benchmark package.
- `fixtures/psion/benchmarks/psion_benchmark_label_generation_receipt_set_v1.json`
  contains the canonical label-generation receipt for that package.
- `fixtures/psion/acceptance/psion_acceptance_matrix_v1.json` now binds the
  memorization-versus-reasoning requirements directly to
  `psion_memorization_reasoning_benchmark_v1` instead of hiding them under the
  generic held-out technical reasoning family.
- `crates/psionic-train/src/psion_benchmark_packages.rs` owns the typed
  memorization-versus-reasoning item payloads inside the shared package
  contract.
- `crates/psionic-train/src/psion_acceptance_matrix.rs` owns the direct
  benchmark-artifact binding used by the acceptance matrix.

## Package Identity

The first dedicated package is:

- package id: `psion_memorization_reasoning_benchmark_v1`
- package family: `memorization_versus_reasoning`
- package digest:
  `599e61d8c6bb7f599042672aaef51b53f4401856437be2a1e688276134cb24a3`

## Typed Coverage

The canonical package now covers five probe kinds explicitly:

- altered-constraint recombination
- unfamiliar design synthesis
- historical analogy transfer
- paraphrase variation
- spec-adjacent edge case

Each probe now preserves:

- `seed_fact_ref`
- `perturbation_ref`
- `probe_kind`
- `expected_transfer`
- `recombination_required`
- `surface_form_shift_required`
- `verbatim_recall_forbidden`

This keeps the package focused on transfer and recombination instead of
accepting answers that merely replay a familiar phrase with high confidence.

## Labels And Receipts

The memorization-versus-reasoning package stays exact on the label side:

- the package receipt stays on the shared `PsionBenchmarkPackageReceipt`
  contract
- the label-generation receipt binds deterministic exact truth for each probe
  item
- the acceptance-ready evidence now lands under the dedicated
  `memorization_versus_reasoning` family instead of the broader held-out
  reasoning family

## Contamination Attachment

The package carries contamination attachment through the committed
contamination-input bundle:

- benchmark-visible source ids remain explicit
- held-out and training-excluded review inputs remain explicit
- the near-duplicate review reference remains explicit
- the package still only counts as green at promotion time when the phase gate
  carries a clean contamination review

## Acceptance Binding

`Psion` acceptance-matrix `v1` now binds the memorization-versus-reasoning
gate directly to the concrete benchmark package artifact above.

The package is required from `sft_promotion` onward:

- `sft_promotion`: minimum `7800` bps
- `internal_serving`: minimum `7900` bps
- `trusted_cluster_scale_up`: minimum `8000` bps

This keeps recombination failure visible in promotion decisions instead of
leaving it buried inside generic held-out benchmark prose.
