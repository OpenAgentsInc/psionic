# TAS-166 Owned Transformer Stack Audit

`TAS-166` freezes the boundary that Psionic now has a real canonical owned
Transformer stack for the article-equivalence closure wave.

This issue does not close the final article model artifact, weight lineage,
reference-linear exactness, fast-route promotion, frontend breadth,
interpreter breadth, benchmark parity, no-spill single-run closure,
clean-room weight causality, cross-machine reproducibility, route minimality,
or the final article-equivalence verdict. It makes the current stack boundary
machine-legible so later work cannot quietly drift back into mixed ownership
or a weaker proof target.

## What Landed

- one committed eval audit at
  `fixtures/tassadar/reports/tassadar_owned_transformer_stack_audit_report.json`
- one committed research summary at
  `fixtures/tassadar/reports/tassadar_owned_transformer_stack_audit_summary.json`
- one boundary-doc update in
  `docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md`
- one audit note at
  `docs/audits/2026-03-20-tassadar-owned-transformer-stack-audit.md`

## Stack Boundary

The audit now says the real owned stack exists across these surfaces:

- `psionic-transformer` owns reusable attention, block, and encoder-decoder
  architecture
- `psionic-models` owns the canonical article wrapper above that extracted
  architecture
- `psionic-train` owns one bounded article-Transformer training lane rooted in
  the canonical wrapper
- `psionic-runtime` owns one runtime evidence and proof-bundle lane rooted in
  the same canonical route

The audit also keeps the rest of the boundary explicit:

- `crates/psionic-models/src/tassadar.rs` remains a fixture-backed legacy lane
- `crates/psionic-models/src/tassadar_executor_transformer.rs` remains a
  research and comparison lane
- `psionic-core`, `psionic-array`, and `psionic-nn` remain lower substrate
  rather than article-route proof by themselves

## Extraction Statement

The audit records the `psionic-transformer` extraction consequences directly:

- reusable Transformer architecture now lives in `psionic-transformer`
- the canonical article wrapper and trace-hook/model surface still live in
  `psionic-models`
- lower tensor, array, and primitive-layer substrate still live in
  `psionic-core`, `psionic-array`, and `psionic-nn`

That split is now treated as the canonical owned-stack truth rather than
generic substrate overlap.

## What It Still Does Not Prove

The audit stays explicit that the remaining blocker categories are still open:

- frontend/compiler envelope and corpus widening
- interpreter breadth and article interpreter suite closure
- final Transformer-backed artifact, weight lineage, parity harness, and
  reference-linear exactness
- canonical fast-route selection, implementation, exactness, and throughput
- article-demo and benchmark parity
- single-run no-spill million-step closure
- clean-room weight causality, KV-cache discipline, reproducibility,
  route-minimality, and final claim checking

## Closure-Gate Tie

The new audit ties directly into the final article-equivalence acceptance gate
through `TAS-166`.

Current bounded truth:

- `TAS-166` is now green inside the acceptance gate
- the gate itself remains `blocked`
- article equivalence itself remains red

## Validation

- `cargo test -p psionic-eval owned_transformer_stack_audit -- --nocapture`
- `cargo test -p psionic-research owned_transformer_stack_audit_summary -- --nocapture`
- `cargo test -p psionic-eval article_equivalence_blocker_matrix -- --nocapture`
- `cargo test -p psionic-eval article_equivalence_acceptance_gate -- --nocapture`

## Claim Boundary

This issue closes only the canonical "actual owned Transformer stack now
exists" audit boundary. It does not imply that the canonical article artifact,
weight lineage, exactness proof, benchmark parity, final publication verdict,
or the final article-equivalence claim are already complete.

## Audit Statement

Psionic now has one canonical machine-readable owned Transformer stack audit
that says the extracted multi-crate stack is real, keeps fixture-backed and
research-only non-canonical lanes explicit, records the remaining open blocker
set directly, and leaves the final article-equivalence verdict blocked.
