# `TAS-171A` Reference-Linear Transformer Exactness Gate Audit

## Scope

Certify that the owned Transformer-backed reference-linear route is exact on
the full declared article workload family before any fast-route promotion work
starts.

## What Changed

- `psionic-eval` now commits the exactness gate at
  `fixtures/tassadar/reports/tassadar_article_transformer_reference_linear_exactness_gate_report.json`
- `psionic-research` now mirrors the operator-readable summary at
  `fixtures/tassadar/reports/tassadar_article_transformer_reference_linear_exactness_summary.json`
- `scripts/check-tassadar-article-transformer-reference-linear-exactness.sh`
  now acts as the dedicated gate checker

## Frozen Facts

- all 13 declared article cases are now explicitly classified as `exact`,
  `mismatch`, or `refused` on the Transformer-backed reference-linear route
- the current committed gate is green with 13 exact cases, 0 mismatch cases,
  and 0 refused cases
- the 3 canonical direct-proof cases remain separately bound to the committed
  direct model-weight proof report and trained lineage contract
- the historical fixture model remains visible only as the trusted baseline for
  exactness comparison, not as the proving route for the current exactness gate

## Verdict

The repo now has one machine-readable gate proving that the owned
Transformer-backed reference-linear route is exact on the full declared
article-family workload set.

It does not yet close anti-memorization, contamination independence,
fast-route promotion, benchmark parity, no-spill single-run closure, or the
final article-equivalence verdict.
