# TAS-185A Audit

The canonical article claim route now has one explicit route-minimality audit
and one explicit publication verdict.

That audit is intentionally narrower than a generic "the whole serving stack is
the claim" statement. It is bound to the direct deterministic `HullCache`
runtime path on the canonical article model, the already-declared no-resume
single-run boundary, the interpreter-ownership verdict, the KV-cache and
activation-state discipline verdict, and the declared cross-machine
reproducibility matrix.

The public repo now says, machine-readably:

- the canonical public claim route is the direct `HullCache` runtime lane
- checkpoint restore, spill or tape continuation, persisted continuation, and
  hidden re-entry are excluded from that claim route
- hidden host substitution, helper mediation, and route drift are excluded from
  that claim route
- planner-owned and hybrid orchestration lanes remain outside the public claim
  even when they exist as operator-side surfaces
- the operator verdict is green while the public verdict remains explicitly
  suppressed until `TAS-186`

This is stronger than saying the route merely works. It freezes which route is
the claim, which orchestration layers are not the claim, and which public
posture is still blocked.

It is still not the final article-equivalence verdict.

The audit does not yet say:

- that the final article-equivalence checker is green
- that the remaining blocker-matrix open row is gone
- that the public claim can widen past the bounded article envelope

Canonical artifacts for this tranche:

- `fixtures/tassadar/reports/tassadar_article_route_minimality_audit_report.json`
- `fixtures/tassadar/reports/tassadar_article_route_minimality_audit_summary.json`
- `fixtures/tassadar/reports/tassadar_article_route_minimality_publication_verdict.json`
- `scripts/check-tassadar-article-route-minimality-audit.sh`
