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

`TAS-166` now freezes the stronger distinction between "the actual owned
Transformer stack exists" and "final article equivalence is proven." The real
owned stack now spans reusable attention/block/encoder-decoder architecture in
`psionic-transformer`, the canonical article wrapper in `psionic-models`, the
bounded training lane in `psionic-train`, and the runtime evidence lane in
`psionic-runtime`. The older fixture-backed lane in
`crates/psionic-models/src/tassadar.rs` and the older
`crates/psionic-models/src/tassadar_executor_transformer.rs` comparison lane
remain explicit non-canonical surfaces, while `psionic-core`,
`psionic-array`, and `psionic-nn` remain lower substrate rather than
article-route proof by themselves.

`TAS-167` now adds the runtime-owned machine-step schema for the canonical
article trace domain in
`crates/psionic-runtime/src/tassadar_article_trace_schema.rs`
and binds that schema directly to the shared trace tokenizer in
`crates/psionic-models/src/tassadar_sequence.rs`
plus the canonical article wrapper in
`crates/psionic-models/src/tassadar_article_transformer.rs`.
That split keeps prompt/trace boundary truth, stack/local/memory channel truth,
and terminal halt-marker truth explicit at the runtime boundary while leaving
tokenization and source/target token batching in the model boundary instead of
collapsing those responsibilities back into the legacy
`tassadar_executor_transformer.rs` comparison lane.

`TAS-167A` now adds the first explicit prompt, tokenization, and
representation invariance gate on top of that trace-bound route.
`psionic-runtime` now owns one explicit prompt-field surface plus local-slot
remap and unreachable-suffix helpers in
`crates/psionic-runtime/src/tassadar_article_representation_invariance.rs`,
while `psionic-models` now owns symbolic retokenization and prompt/target
symbolic recomposition helpers in
`crates/psionic-models/src/tassadar_sequence.rs`.
That split lets the repo prove exact-trace stability for whitespace, prompt
field-order, and dead-code layout perturbations, while keeping local-renaming
cases explicitly representation-sensitive but canonically equivalent instead of
pretending every semantic-preserving perturbation must leave the raw trace
identical. The machine-readable gate also keeps bounded suppressions explicit
when an article case exceeds the current trace-domain reference-model position
window rather than widening the support boundary silently.

## Route Requirement

Any canonical article-equivalence Transformer route must use this boundary.
Later issues may extend the implementations inside these owners, but they must
not redefine the ownership split or route around it.
