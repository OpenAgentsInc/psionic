# Psion Forge Eval Pack Manifests

> Status: canonical `#818` eval-pack publication surface, written 2026-04-02
> after landing the first benchmark-pack and judge-pack manifests for Forge
> consumption.

This document records the first narrow Psionic-owned manifest surface for
Forge-facing benchmark and judge packs.

It does not create a second eval runtime.

It publishes selectors over artifacts Psionic already owns:

- the canonical benchmark catalog
- the canonical benchmark receipt set
- the grader interfaces already attached to those benchmark packages

## Canonical Artifacts

- `crates/psionic-train/src/psion_forge_eval_pack_manifests.rs`
  - owns the typed benchmark-pack and judge-pack manifest contracts
- `crates/psionic-train/examples/psion_forge_eval_pack_manifest_fixtures.rs`
  - regenerates the committed manifest fixtures from the canonical benchmark
    catalog and receipt set
- `fixtures/psion/packs/psion_forge_benchmark_pack_manifest_v1.json`
  - first published benchmark-pack manifest
- `fixtures/psion/packs/psion_forge_judge_pack_manifest_v1.json`
  - first published judge-pack manifest

The stable schema versions are:

- `psion.forge_benchmark_pack_manifest.v1`
- `psion.forge_judge_pack_manifest.v1`

## Benchmark-Pack Manifest

The benchmark-pack manifest publishes the current bounded Psion benchmark set
as one typed pack.

Each package entry carries:

- `package_id`
- `package_family`
- optional `acceptance_family`
- generic `benchmark_ref` and `benchmark_version`
- stable `benchmark_digest`
- the bounded review `receipt_id` and `receipt_digest`
- `phase`
- `item_count`
- minimal metric selectors from the receipt

That gives Forge a stable mountable object over benchmark packages without
hardcoding fixture paths or reverse-engineering receipt files.

## Judge-Pack Manifest

The judge-pack manifest publishes the grader interfaces already bound to the
canonical benchmark catalog.

Each judge entry carries:

- `judge_id`
- `judge_kind`
- bound `package_ids`
- bound `package_families`
- optional `acceptance_families`
- type-specific selectors such as:
  - `rubric_ref`
  - `label_namespace`
  - `accepted_labels`
  - `expected_route`
  - `accepted_reason_codes`

This is the first honest "judge pack" surface because those selectors already
exist inside Psionic benchmark contracts today.

## Source Refs And Claim Boundary

Both manifest families carry explicit source artifacts with repo-local paths
and SHA256 digests.

That matters because Forge should reference Psionic-owned eval artifacts by
stable identity instead of scraping arbitrary file names or inventing app-owned
judge schemas.

The current claim boundary is intentionally narrow:

- these manifests do publish stable pack ids and source refs
- they do not publish campaign admission
- they do not publish payout authority
- they do not imply arbitrary remote rubric hosting
- they do not replace the underlying benchmark package or receipt contracts
