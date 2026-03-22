# PSION Plugin-Conditioned Dataset Bundle

> Status: canonical `PSION_PLUGIN-5` dataset-bundle contract for the first
> host-native plugin-conditioned convergence tranche, written 2026-03-22 after
> landing the first repo-owned dataset builder on top of the committed
> derivation bundle.

This document freezes the first dataset artifact bundle built from canonical
plugin-training records.

It is intentionally small and explicit.

The point is not volume.

The point is to emit one stable dataset identity, one split contract, and one
machine-checkable held-out boundary that later training and benchmark work can
reuse directly.

## Canonical Artifacts

- `docs/PSION_PLUGIN_CONDITIONED_DATASET.md` is the canonical human-readable
  contract.
- `crates/psionic-data/src/psion_plugin_conditioned_dataset.rs` owns the
  typed dataset bundle and split validation.
- `crates/psionic-data/examples/psion_plugin_conditioned_dataset.rs` writes
  the canonical dataset bundle.
- `fixtures/psion/plugins/datasets/psion_plugin_conditioned_dataset_v1/` carries
  the first committed dataset artifact.

The stable bundle schema version is
`psionic.psion.plugin_conditioned_dataset_bundle.v1`.

## Current Split Policy

The first dataset bundle uses one intentionally conservative isolation rule:

- training and held-out records must be disjoint by `workflow_case_id`

That means the same normalized workflow case cannot appear in both splits,
even if it was observed through different controller surfaces.

The current v1 build keeps:

- one `train` split
- one `held_out` split

This is enough for the first bounded SFT and evaluation lane.

## Preserved Labels

The dataset bundle keeps the following labels visible per split:

- controller-surface counts
- plugin-class counts
- route-label counts
- outcome-label counts

That keeps later model and benchmark work from treating the derived records as
anonymous prompt/response examples.

## Stable Identity

The bundle now carries:

- one stable dataset ref
- one immutable version
- one stable `dataset_ref@version` identity
- one source derivation bundle ref and digest

That gives later training stages one direct citation surface instead of
pointing at a loose folder of JSON records.
