# TAS-161 Attention Primitive And Mask Closure

`TAS-161` lands the owned scaled dot-product attention primitive and masking
path inside `psionic-transformer`.

This issue does not close the full article-equivalent Transformer stack. It
freezes the reusable attention math, mask composition, and probability-trace
export that later article-route work must build on.

## What Landed

- one reusable attention implementation in
  `crates/psionic-transformer/src/attention.rs`
- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_attention_primitive_mask_closure_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_attention_primitive_mask_closure_summary.json`
- one audit note at
  `docs/audits/2026-03-20-tassadar-attention-primitive-mask-closure.md`

## Primitive Scope

The owned reusable path now covers:

- `QK^T / sqrt(d_k)`
- numerically stable softmax
- causal masks
- padding masks
- combined masks
- attention-probability trace export

The reusable primitive lives in `psionic-transformer`, uses `psionic-core` for
tensor-spec and host-data export, and uses the bounded `psionic-array` CPU
surface for matrix products.

## Boundary Statement

The crate boundary is explicit:

- `psionic-transformer` now defines the owned reusable symbol
  `scaled_dot_product_attention`
- `psionic-transformer` now depends directly on `psionic-core` and
  `psionic-array`
- `psionic-transformer` still does not depend directly on `psionic-models` or
  `psionic-runtime`
- `psionic-models` does not define or re-own the reusable attention symbol

This keeps reusable attention logic at the canonical Transformer layer instead
of letting it drift back into mixed model-policy crates.

## Closure-Gate Tie

The new artifact ties directly into the final article-equivalence acceptance
gate through `TAS-161`.

Current bounded truth:

- `TAS-161` is now green inside the acceptance gate
- the gate itself remains `blocked`
- article equivalence itself remains red

## Validation

- `cargo test -p psionic-transformer -- --nocapture`
- `cargo test -p psionic-eval attention_primitive_mask_closure -- --nocapture`
- `cargo test -p psionic-research attention_primitive_mask_closure_summary -- --nocapture`

## Claim Boundary

This issue closes only the owned attention primitive and mask path. It does
not imply that the full encoder-decoder stack, the canonical article artifact,
the reference-linear article route, or the final article-equivalence proof are
already complete.

## Audit Statement

Psionic now has one canonical owned scaled dot-product attention, masking, and
probability-trace primitive in `psionic-transformer`, tied directly to the
final article-equivalence gate, while the overall article-equivalence verdict
remains blocked.
