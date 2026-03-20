# `TAS-169A` Article Transformer Weight Lineage Contract Audit

## Scope

Freeze the provenance and artifact contract around the first real trained
trace-bound article-Transformer weights without widening the public
article-equivalence claim.

## What Changed

- `psionic-models` now points the trained trace-bound article route at one
  committed lineage manifest:
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0_lineage_contract.json`
- `psionic-eval` now audits that manifest in
  `fixtures/tassadar/reports/tassadar_article_transformer_weight_lineage_report.json`
- `psionic-research` now mirrors the operator-readable summary in
  `fixtures/tassadar/reports/tassadar_article_transformer_weight_lineage_summary.json`

## Frozen Facts

- exact workload set:
  `hungarian_matching` for training and
  `micro_wasm_kernel`, `branch_heavy_kernel`, `memory_heavy_kernel` for held
  out
- training-config snapshot:
  the committed trace-bound article wrapper, `32`-token target window,
  `logits_projection_bias` as the only trainable surface, and the same bounded
  optimizer/scheduler configuration recorded in the `TAS-169` bundle
- source inventory:
  model, Transformer, train, and runtime source refs plus file digests
- artifact inventory:
  evidence bundle, base descriptor, base weights, produced descriptor, and
  produced weights plus file digests
- checkpoint lineage:
  the committed checkpoint digest chain and trained/restored parameter digest
  parity

## Verdict

The repo now separates "there is a first real trained article artifact" from
"that artifact has a frozen, challengeable provenance contract." This closes
the lineage bar for the bounded trained trace-bound model only.

It does not close reference-linear exactness, fast-route promotion, benchmark
parity, single-run closure, or final article-equivalence green status.
