# Tassadar Article Transformer Stack Boundary

This document is the canonical Transformer stack boundary for the
article-equivalence closure wave.

The final article route is not one monolithic new crate. It is one explicit
boundary spanning existing crates, with `psionic-transformer` as the
architecture anchor.

## Boundary Decision

The canonical owned article route must use this boundary:

- `psionic-core` plus `psionic-array` for tensor and bounded array ops
- `psionic-nn` plus `psionic-transformer` for layer, parameter-state, and
  reusable Transformer-architecture interfaces
- `psionic-models` for the canonical article-model wrapper, weight artifact
  format, and forward-pass-owned trace hook surface
- `psionic-runtime` for replay, trace ABI, proof identity, and receipt hooks

The final article route must not bypass `psionic-transformer` by reintroducing
an ad hoc mixed implementation spread directly across model and runtime crates.

## Ownership Diagram

```text
psionic-core + psionic-array
  -> tensor metadata, layouts, devices, bounded array ops

psionic-nn + psionic-transformer
  -> module state, primitive layers, reusable Transformer architecture

psionic-models
  -> canonical article model artifact, forward-pass hooks, weight bundle format

psionic-runtime
  -> replay, trace ABI, proof identity, and receipt hooks
```

## Required Interfaces

### Tensor and array ops

- owner modules:
  `crates/psionic-core/src/lib.rs`
  `crates/psionic-array/src/lib.rs`
- purpose:
  the article route consumes tensor metadata, layout truth, and bounded array
  execution through these shared surfaces

### Layer and parameter state

- owner modules:
  `crates/psionic-nn/src/lib.rs`
  `crates/psionic-nn/src/layers.rs`
  `crates/psionic-transformer/src/lib.rs`
  `crates/psionic-transformer/src/attention.rs`
  `crates/psionic-transformer/src/blocks.rs`
  `crates/psionic-transformer/src/encoder_decoder.rs`
- purpose:
  primitive layer semantics plus reusable Transformer attention, embeddings,
  feed-forward, residual and norm block composition, encoder-decoder stack
  assembly, masking, and probability-trace export stay here;
  `psionic-transformer` is the architecture anchor

### Model artifact format

- owner modules:
  `crates/psionic-models/src/lib.rs`
  `crates/psionic-models/src/tassadar_article_transformer.rs`
- purpose:
  the canonical article model descriptor, route selection, and later weight
  artifact identity all live in `psionic-models`

### Forward-pass trace hooks

- owner modules:
  `crates/psionic-models/src/tassadar_article_transformer.rs`
  `crates/psionic-runtime/src/tassadar.rs`
- purpose:
  forward-pass-owned trace hooks are emitted by the model boundary and
  serialized into runtime-owned trace ABI surfaces

## Canonical Wrapper Note

The canonical article wrapper is now
`crates/psionic-models/src/tassadar_article_transformer.rs`.

The older
`crates/psionic-models/src/tassadar_executor_transformer.rs`
remains a separate research and comparison lane. It is not the canonical
paper-faithful article route.

### Replay and receipt hooks

- owner modules:
  `crates/psionic-runtime/src/proof.rs`
  `crates/psionic-runtime/src/tassadar.rs`
- purpose:
  replay, proof identity, trace receipts, and execution challenge hooks remain
  runtime-owned

## Dependency Direction

The boundary is backed by real dependency checks:

- `psionic-models` must depend on `psionic-transformer`
- `psionic-transformer` must depend on `psionic-nn`
- `psionic-transformer` must not depend on `psionic-models` or
  `psionic-runtime`
- `psionic-nn` must not depend on `psionic-train`, `psionic-models`, or
  `psionic-runtime`
- `psionic-runtime` must not depend on `psionic-models`
- `psionic-models` must not depend on `psionic-eval` or `psionic-research`

Optimizer and training-facing orchestration can live above this boundary, but
it must not pull the lower-level `psionic-nn` layer substrate back into the
training or model crates. The repo now keeps that interop split in
`psionic-nn-optimizers`.

`TAS-164` now closes the first bounded article-Transformer training lane on
top of this split in
`crates/psionic-train/src/tassadar_article_transformer_training.rs`.
That lane consumes the canonical article wrapper from `psionic-models`,
keeps reusable encoder-decoder architecture ownership in
`psionic-transformer`, and leaves optimizer-loop orchestration in
`psionic-train` rather than reabsorbing Transformer ownership into the train
crate.

`TAS-165` now closes the first runtime receipt lane on top of the same split.
`psionic-models` now emits one canonical
`forward_with_runtime_evidence(...)` entrypoint from
`crates/psionic-models/src/tassadar_article_transformer.rs`, while
`psionic-runtime` now owns the article-Transformer runtime bundle in
`crates/psionic-runtime/src/tassadar_article_transformer_forward_pass.rs`.
That keeps forward-pass trace capture at the model boundary, reusable
encoder-decoder execution in `psionic-transformer`, and runtime-manifest plus
proof-bundle ownership in `psionic-runtime` instead of collapsing those
receipt hooks back into the model crate.

## Route Requirement

Any canonical article-equivalence Transformer route must use this boundary.
Later issues may extend the implementations inside these owners, but they must
not redefine the ownership split or route around it.
