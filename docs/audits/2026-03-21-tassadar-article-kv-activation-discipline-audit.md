# TAS-184A Audit

## Summary

`TAS-184A` closes the KV-cache and activation-state discipline tranche on the
canonical owned article route.

The repo now has one joined eval artifact at
`fixtures/tassadar/reports/tassadar_article_kv_activation_discipline_audit_report.json`
plus the mirrored operator summary at
`fixtures/tassadar/reports/tassadar_article_kv_activation_discipline_audit_summary.json`.

## Evidence

The audit stays tied to committed repo evidence only:

- the committed TAS-184 interpreter-ownership gate
- the current article-equivalence acceptance gate
- the committed reference-linear exactness report for the declared case set
- analytic KV-cache and activation-state accounting over the declared article
  case IDs
- constrained-history sensitivity rows covering moderate window, strict
  window, and mid-decode reset conditions
- explicit acceptable versus non-acceptable carrier boundaries for KV cache,
  residual stream, and attention history

## Claim Boundary

This closes the bounded route's explicit state-discipline verdict only inside
the public article envelope.

The route is now declared `mixed`: decisive behavior remains weight-sensitive
and route-owned, while request-local KV and activation state still carry
same-run execution history inside the admitted forward pass.

It does not imply cross-machine reproducibility closure, route minimality, or
final article-equivalence green status.
