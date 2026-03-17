# Apple Adapter Fixtures

This directory freezes the repo-owned Apple adapter conformance inputs introduced
for issue `#3616`.

## Scope

- `datasets/`: positive and negative JSONL training-data examples, plus the
  first reviewed real-run corpus splits under
  `datasets/psionic_architecture_explainer/`
- `experiments/`: frozen experiment manifests and trend ledgers for the
  first reviewed `Psionic architecture explainer` run iterations, including the
  standalone repo-local reference-overfit contract at
  `experiments/psionic_architecture_explainer_reference_overfit_v1.json`
- `runs/`: committed machine-readable run reports, including the standalone
  repo-local Apple overfit proof at
  `runs/psionic_architecture_explainer_reference_overfit_report.json`
- `packages/`: positive and negative `.fmadapter` inventory fixtures
- `lineage/`: positive and negative OpenAgents lineage payloads

## Important note about payload files

The `.bin` and `.mil` files in this fixture corpus are intentionally small
placeholder payloads. They are not executable Apple-exported weights.

Their job is to freeze:

- file inventory
- metadata shape
- compatibility anchors
- digest behavior
- rejection cases

That keeps the repo lightweight while still giving later Rust parser/writer
tests a stable corpus to validate against.

## Canonical docs

- `docs/APPLE_ADAPTER_DATASET_SPEC.md`
- `docs/APPLE_FMADAPTER_PACKAGE_SPEC.md`
- `docs/APPLE_ADAPTER_LINEAGE_SPEC.md`
- `docs/TRAIN_SYSTEM.md`

## Repo-Local Overfit Evidence

The standalone `psionic` repo now keeps one canonical bounded Apple
benchmark-effectiveness proof entirely in-repo:

- frozen contract:
  `experiments/psionic_architecture_explainer_reference_overfit_v1.json`
- generator:
  `crates/psionic-train/examples/apple_architecture_explainer_reference_overfit.rs`
- machine-readable report:
  `runs/psionic_architecture_explainer_reference_overfit_report.json`

That report is intentionally repo-local reference evidence, not a live
bridge-backed or authority-backed operator receipt.
