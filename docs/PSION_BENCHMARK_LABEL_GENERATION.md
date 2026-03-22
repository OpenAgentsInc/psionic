# Psion Benchmark Label Generation

> Status: canonical `PSION-20` / `#376` benchmark-label-generation contract,
> written 2026-03-22 after landing exact-truth, rubric-version, and
> derived-data-lineage receipts for the main `Psion` benchmark families.

This document freezes the first `Psion` benchmark label-generation contract.

`docs/PSION_BENCHMARK_PACKAGES.md` already standardized package shape, prompt
format, grader interface, contamination inputs, and benchmark receipts. This
document adds the missing next layer: how benchmark labels are actually
produced, versioned, and traced back to reviewed parent sources or generators.

## Canonical Artifacts

- `crates/psionic-train/src/psion_benchmark_label_generation.rs` owns the
  label-generation receipt and receipt-set contracts.
- `crates/psionic-train/examples/psion_benchmark_label_generation_fixtures.rs`
  regenerates the canonical label-generation fixture.
- `fixtures/psion/benchmarks/psion_benchmark_label_generation_receipt_set_v1.json`
  is the canonical receipt set spanning the main benchmark families.
- `fixtures/psion/lifecycle/psion_artifact_lineage_manifest_v1.json` remains
  the canonical benchmark-artifact lineage manifest that the new receipts bind
  to directly.

The stable schema versions are:

- `psion.benchmark_label_generation_receipt.v1`
- `psion.benchmark_label_generation_receipt_set.v1`

## What The Contract Standardizes

The shared contract now freezes:

- one item-level label-generation receipt shape with explicit
  exact-versus-rubric posture
- one package-level receipt that derives whether a benchmark family is
  exact, rubric-backed, or hybrid from the item receipts underneath it
- one versioned label-generation-logic binding per generated benchmark item
- one exact-truth binding surface that admits CPU-reference labels,
  equivalent exact labels, exact route truth, and exact refusal truth
- one rubric-version binding surface for benchmark items that still require
  human judgment
- one derived-data-lineage surface that preserves generated item and label
  digests, parent source ids, parent artifact refs, and generator refs

## Mechanical Enforcement

`psionic-train` now validates that:

- label-generation receipts still match the canonical benchmark package ids,
  package digests, grader ids, and contamination-input digests
- exact-label, exact-route, and exact-refusal items cannot appear without an
  exact-truth binding that matches their declared grader contract
- rubric-backed items cannot appear without a rubric ref and rubric version
  that matches the package grader contract
- generated benchmark item lineage still matches the parent source ids already
  declared on the benchmark package item itself
- the union of item-level parent source ids still matches the benchmark
  artifact row in the canonical source-lineage manifest
- mixed exact-plus-rubric packages are surfaced honestly as `hybrid` instead
  of being flattened into one fake grader posture

## Canonical Coverage

The committed receipt set proves the contract across the main benchmark
families:

- architecture reasoning stays rubric-backed with explicit rubric versioning
- normative spec reading stays exact with equivalent exact truth bindings
- engineering spec interpretation now proves the hybrid posture with one
  rubric-backed item and one CPU-reference exact-label item
- memorization-versus-reasoning stays exact with deterministic label truth
- route evaluation stays exact with route-policy truth
- refusal evaluation stays exact with refusal-policy truth

## Why This Matters

This closes the label-credibility gap in the learned-model benchmark lane:

- exact benchmark families can now point to CPU-reference or equivalent exact
  truth instead of relying on an unexplained label blob
- rubric-backed benchmark families now preserve rubric version and reviewer
  guidance instead of pretending human judgment is timeless
- hybrid packages can now be expressed honestly when one family needs both
  exact and rubric-backed item generation
- benchmark invalidation review can now follow the same source-lifecycle graph
  as tokenizer, corpus, SFT, and checkpoint artifacts because derived
  benchmark items and labels keep explicit parent-source lineage
