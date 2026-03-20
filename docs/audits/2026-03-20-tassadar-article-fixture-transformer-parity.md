# `TAS-170` Article Fixture-to-Transformer Parity Audit

## Scope

Certify that the committed trained trace-bound article-Transformer wrapper can
inherit the old fixture lane's declared bounded truth on the canonical article
corpus without pretending the Transformer route already owns the direct proof.

## What Changed

- `psionic-eval` now commits the parity certificate at
  `fixtures/tassadar/reports/tassadar_article_fixture_transformer_parity_report.json`
- `psionic-research` now mirrors the operator-readable summary at
  `fixtures/tassadar/reports/tassadar_article_fixture_transformer_parity_summary.json`
- `psionic-serve` now publishes the bounded served certificate at
  `fixtures/tassadar/reports/tassadar_article_transformer_replacement_publication.json`

## Frozen Facts

- all 13 canonical article cases remain routeable on the fixture baseline and
  on the Transformer trace-domain wrapper
- every declared case keeps exact trace parity after the fixture execution is
  roundtripped through the trained trace-bound Transformer wrapper
- every declared case keeps exact outputs, final locals, final memory, final
  stack, and halt reason after that roundtrip
- the 4 cases that fit the current model window still bind to the committed
  trained model artifact and deterministic forward-pass evidence lane

## Verdict

The repo now has one machine-readable certificate that the owned
Transformer-backed article route can stand in for the old fixture lane as the
bounded truth carrier on the declared article workload set.

It does not yet close direct no-tool proof ownership, reference-linear exact
model behavior from weights alone, fast-route promotion, benchmark parity, or
final article-equivalence green status.
