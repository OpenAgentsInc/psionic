# TAS-165 Article Transformer Forward-Pass Closure

`TAS-165` lands the canonical runtime evidence lane for one owned
article-Transformer forward pass.

This issue does not close final article trace vocabulary, artifact-backed
weight production, reference-linear exactness proof, fast-route promotion,
benchmark parity, or the final article-equivalence route. It freezes the
runtime-manifest and proof-bundle discipline that later closure work must
consume.

## What Landed

- one runtime-owned forward-pass receipt module in
  `crates/psionic-runtime/src/tassadar_article_transformer_forward_pass.rs`
- one model-side evidence entrypoint in
  `crates/psionic-models/src/tassadar_article_transformer.rs`
- one committed runtime evidence bundle at
  `fixtures/tassadar/runs/tassadar_article_transformer_forward_pass_v1/article_transformer_forward_pass_evidence_bundle.json`
- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_article_transformer_forward_pass_closure_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_article_transformer_forward_pass_closure_summary.json`
- one audit note at
  `docs/audits/2026-03-20-tassadar-article-transformer-forward-pass-closure.md`

## Runtime Scope

The owned forward-pass receipt lane now covers:

- digest-bound model descriptor and bounded parameter-bundle identity
- replay-stable run-config capture over source tokens, target tokens, shapes,
  execution mode, and environment refs
- forward-pass-owned attention-trace channel summaries over encoder,
  decoder-self, and decoder-cross attention
- greedy decode receipts over the emitted logits tensor
- deterministic replay receipts over a second forward pass on the same owned
  route
- checkpoint-lineage bindings carried from the bounded TAS-164 training lane
- runtime-manifest and proof-bundle construction on top of those facts

## Boundary Statement

The route boundary stays explicit:

- `psionic-transformer` remains the reusable encoder-decoder forward-pass owner
- `psionic-models` emits one canonical runtime-evidence entrypoint on top of
  that owned stack
- `psionic-runtime` owns the receipt bundle, runtime-manifest bindings, and
  proof-bundle bindings
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md` now records that split
  directly

This keeps forward-pass trace capture at the model boundary while preserving
runtime ownership of the machine-legible receipt substrate.

## Closure-Gate Tie

The new artifact ties directly into the final article-equivalence acceptance
gate through `TAS-165`.

Current bounded truth:

- `TAS-165` is now green inside the acceptance gate
- the gate itself remains `blocked`
- article equivalence itself remains red

## Validation

- `cargo test -p psionic-runtime tassadar_article_transformer_forward_pass -- --nocapture`
- `cargo test -p psionic-models tassadar_article_transformer -- --nocapture`
- `cargo test -p psionic-eval article_transformer_forward_pass_evidence -- --nocapture`
- `cargo test -p psionic-eval article_transformer_forward_pass_closure -- --nocapture`
- `cargo test -p psionic-research article_transformer_forward_pass_closure_summary -- --nocapture`

## Claim Boundary

This issue closes only the canonical runtime evidence lane for one
article-Transformer forward pass. It does not imply that final article trace
vocabulary closure, artifact-backed weights, final exactness proof,
benchmark parity, or the final article-equivalence verdict are already
complete.

## Audit Statement

Psionic now has one canonical article-Transformer runtime evidence lane with
model identity, run-config capture, trace summaries, decode receipts,
deterministic replay receipts, and checkpoint lineage bound cleanly into
`RuntimeManifest` and `ExecutionProofBundle`, while the overall
article-equivalence verdict remains blocked.
