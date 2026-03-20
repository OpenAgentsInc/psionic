# TAS-167 Article Trace Vocabulary Binding Audit

`TAS-167` closes the owned-route trace vocabulary and channel-binding gap for
the canonical article Transformer stack.

The landed boundary is explicit:

- `psionic-runtime` now owns the machine-step schema and prompt/trace/halt
  contract in
  `crates/psionic-runtime/src/tassadar_article_trace_schema.rs`
- `psionic-models` now owns the shared tokenizer and typed trace decode in
  `crates/psionic-models/src/tassadar_sequence.rs`
- `psionic-models` now binds the canonical article wrapper directly to that
  runtime-owned schema in
  `crates/psionic-models/src/tassadar_article_transformer.rs`

This means the owned article route can now:

- consume one typed program plus execution pair and encode it into the shared
  prompt-prefix plus append-only trace-suffix token split
- bind stack, locals, memory, instruction, event, and halt channels directly
  against one runtime-owned schema instead of ad hoc model-only assumptions
- reconstruct one typed `TassadarProgram` and `TassadarExecution` pair back
  from the shared token domain without drift on an article-class case

The committed machine-readable artifacts are:

- `fixtures/tassadar/reports/tassadar_article_trace_vocabulary_binding_report.json`
- `fixtures/tassadar/reports/tassadar_article_trace_vocabulary_binding_summary.json`

This closure is still bounded.

It does not prove:

- prompt/tokenization/representation invariance
- artifact-backed weight identity or lineage
- reference-linear exactness on the Transformer-backed route
- fast-route promotion
- benchmark parity
- final article-equivalence green status

Targeted validation for this tranche:

- `cargo test -p psionic-runtime tassadar_article_trace_schema -- --nocapture`
- `cargo test -p psionic-models tassadar_sequence -- --nocapture`
- `cargo test -p psionic-models tassadar_article_transformer -- --nocapture`
- `cargo test -p psionic-eval article_trace_vocabulary_binding -- --nocapture`
- `cargo test -p psionic-research article_trace_vocabulary_binding_summary -- --nocapture`
- `cargo run -p psionic-eval --example tassadar_article_trace_vocabulary_binding_report`
- `cargo run -p psionic-research --example tassadar_article_trace_vocabulary_binding_summary`
