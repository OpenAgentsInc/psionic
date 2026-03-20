# TAS-163 Article Transformer Model Closure

`TAS-163` lands the canonical paper-faithful article-Transformer model path.

This issue does not close the article trace vocabulary, weight-artifact
lineage, training recipe, replay receipt, exactness, benchmark parity, or the
final article-equivalence route. It freezes the owned encoder-decoder model
path that later closure work must consume.

## What Landed

- one reusable encoder-decoder Transformer stack in
  `crates/psionic-transformer/src/encoder_decoder.rs`
- one canonical article wrapper in
  `crates/psionic-models/src/tassadar_article_transformer.rs`
- one committed eval artifact at
  `fixtures/tassadar/reports/tassadar_article_transformer_model_closure_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_article_transformer_model_closure_summary.json`
- one audit note at
  `docs/audits/2026-03-20-tassadar-article-transformer-model-closure.md`

## Model Scope

The owned article-model path now covers:

- one explicit `Attention Is All You Need` paper reference at the model
  descriptor boundary
- encoder stack execution
- masked decoder self-attention
- cross-attention over encoder states
- decoder-to-logits projection
- explicit unshared, decoder-tied, and fully shared source/target/output
  weight-sharing modes

These pieces now live on one canonical route split across
`psionic-transformer` and `psionic-models` instead of remaining implicit or
living only in the older executor-transformer scaffold.

## Boundary Statement

The route boundary is explicit:

- `psionic-transformer` now owns the reusable encoder-decoder stack
- `psionic-models` now owns the canonical article wrapper
- `psionic-models/src/tassadar_executor_transformer.rs` remains a separate
  research and comparison lane rather than the canonical article route
- `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md` now points at the new
  canonical wrapper and encoder-decoder owner modules

This keeps paper-faithful architecture logic at the Transformer layer while
keeping the canonical article-model selection at the model boundary.

## Closure-Gate Tie

The new artifact ties directly into the final article-equivalence acceptance
gate through `TAS-163`.

Current bounded truth:

- `TAS-163` is now green inside the acceptance gate
- the gate itself remains `blocked`
- article equivalence itself remains red

## Validation

- `cargo test -p psionic-transformer -- --nocapture`
- `cargo test -p psionic-models tassadar_article_transformer -- --nocapture`
- `cargo test -p psionic-eval article_transformer_model_closure -- --nocapture`
- `cargo test -p psionic-research article_transformer_model_closure_summary -- --nocapture`

## Claim Boundary

This issue closes only the canonical paper-faithful article model path. It
does not imply that article trace vocabulary closure, artifact-backed weight
lineage, trained article weights, replay receipts, reference-linear proof
closure, or the final article-equivalence verdict are already complete.

## Audit Statement

Psionic now has one canonical paper-faithful article-Transformer model path,
split cleanly between `psionic-transformer` and `psionic-models`, tied
directly to the final article-equivalence gate, while the overall
article-equivalence verdict remains blocked.
