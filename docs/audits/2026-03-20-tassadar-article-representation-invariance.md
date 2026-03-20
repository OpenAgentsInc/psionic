# TAS-167A Article Representation Invariance Audit

`TAS-167A` closes the first explicit prompt, tokenization, and representation
invariance gate for the canonical owned article route.

The landed split is explicit:

- `psionic-runtime` now owns the prompt-field surface plus local-slot remap,
  inverse-remap, and unreachable-suffix helpers in
  `crates/psionic-runtime/src/tassadar_article_representation_invariance.rs`
- `psionic-models` now owns symbolic retokenization and prompt/target
  symbolic-text recomposition helpers in
  `crates/psionic-models/src/tassadar_sequence.rs`
- `psionic-eval` now owns the machine-readable invariance gate in
  `fixtures/tassadar/reports/tassadar_article_representation_invariance_gate_report.json`
- `psionic-research` now owns the disclosure-safe operator summary in
  `fixtures/tassadar/reports/tassadar_article_representation_invariance_gate_summary.json`

This means the owned article route now:

- keeps exact behavior under whitespace and formatting perturbations over the
  shared symbolic token text surface
- keeps exact behavior when prompt and target symbolic token text are
  re-encoded independently and recomposed at the declared boundary
- treats semantically irrelevant prompt-field ordering as canonical runtime
  materialization rather than a brittle input surface
- keeps local-slot renaming explicit as representation-sensitive while still
  proving canonical semantic equivalence after inverse remapping
- keeps dead-code suffix IR layout perturbations exact instead of conflating
  prompt-only layout changes with execution changes
- keeps long-horizon position-window suppressions machine-legible rather than
  pretending the bounded trace-domain reference model already supports every
  article-class trace length

This closure is still bounded.

It does not prove:

- artifact-backed weight identity or lineage
- reference-linear exactness on the Transformer-backed route
- fast-route promotion
- benchmark parity
- final article-equivalence green status

Targeted validation for this tranche:

- `cargo test -p psionic-runtime tassadar_article_representation_invariance -- --nocapture`
- `cargo test -p psionic-models tassadar_sequence -- --nocapture`
- `cargo test -p psionic-eval article_representation_invariance_gate -- --nocapture`
- `cargo test -p psionic-research article_representation_invariance_gate_summary -- --nocapture`
- `cargo run -p psionic-eval --example tassadar_article_representation_invariance_gate_report`
- `cargo run -p psionic-research --example tassadar_article_representation_invariance_gate_summary`
