# Psion Actual Pretraining Data Bundle

> Status: canonical actual-lane data authority, written 2026-04-02 after
> freezing one repo-owned transformation, filtering, deduplication, replay,
> and mixture contract for `psion_actual_pretraining_v1`.

This document freezes the canonical data authority consumed by the actual
`Psion` pretraining lane.

It does not create a detached data-research program. It binds CS336 A4-shaped
data work directly into the frozen actual lane so recipe changes, replay, and
mixture authority stay machine-legible and repeatable.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_data_bundle.rs` owns the
  typed data-bundle contract.
- `crates/psionic-train/examples/psion_actual_pretraining_data_bundle_fixtures.rs`
  regenerates the committed fixture from repo-owned source, tokenizer,
  benchmark, and sampling manifests.
- `fixtures/psion/pretrain/psion_actual_pretraining_data_bundle_v1.json` is the
  canonical machine-readable data bundle consumed by the actual lane.
- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md` freezes the recipe and
  topology/storage bundle this data authority feeds.

The stable schema version is:

- `psion.actual_pretraining_data_bundle.v1`

## Frozen Data Authority

The actual-lane data bundle fixes:

- one ordered transformation path from admission review through tokenized corpus
  build and sampling-policy freeze
- one filter authority that distinguishes admitted training sources,
  tokenizer-only sources, held-out eval sources, and rejected sources
- one dedup authority that blocks repetitive regions and near-duplicate leakage
  into train or benchmark publication
- one frozen production mixture with bounded family weights, source caps, and
  content-class reporting
- one replay authority with deterministic shuffle, fixed seed, packing policy,
  max sequence length, and explicit split shard ids
- one recipe-change eval package that must cover architecture reasoning,
  normative-spec reading, engineering-spec interpretation, and
  memorization-versus-reasoning checks

The current canonical bundle remains tied to:

- dataset identity: `psion_corpus_tokenized@v1`
- sampling policy: `psion_pretrain_mix@v1`
- lane id: `psion_actual_pretraining_v1`

## What This Prevents

Without one committed data bundle, the actual lane would still depend on loose
claims about what got filtered, how dedup happened, which mixture counts as
production, and how recipe changes are evaluated.

This document prevents that drift by making the actual lane consume one
validated data contract instead of reconstructing data authority from scattered
manifests at operator time.

## Claim Boundary

This bundle is the actual-lane data authority. It is not a general data
research lane, an unfrozen curriculum port, or a claim that broader data
infrastructure hardening is already finished.
