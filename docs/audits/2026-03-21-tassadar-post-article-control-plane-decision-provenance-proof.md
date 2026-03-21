# TAS-188A Audit

The public repo now has one machine-readable post-article control-plane
decision-provenance proof.

That proof closes a different tranche than `TAS-188`. `TAS-188` proved state
ownership and semantic preservation for continuation mechanics. This new proof
adds the positive control-plane claim: branch, retry, and stop decisions on
the rebased route are now machine-readably bound to model outputs, canonical
route identity, and the bridge machine identity tuple.

The public repo now says, machine-readably:

- branch, retry, and stop decisions are individually bound to model outputs,
  canonical route identity, and the bridge machine identity
- the selected control-trace determinism class is `strict_deterministic`
- the current equivalent-choice relation is the singleton exact-trace relation
  with zero admissible control divergence
- typed failure classes, logical-time semantics, information boundaries,
  training-versus-inference boundaries, hidden-state closure, and observer
  acceptance requirements are all frozen as explicit objects
- latency, cost, queueing, scheduling, cache hits, helper substitution, and
  fast-route fallback remain blocked as hidden control channels

This is stronger than saying the repo merely forbids host control in prose. It
freezes one proof-carrying control contract for the current rebased route.

It is still not the final rebased universality claim.

The audit does not say:

- that the final direct-versus-resumable carrier split is already published
- that the rebased Turing-completeness claim is already admitted on the
  canonical route
- that weighted plugin control is already in scope
- that served/public universality is already allowed
- that arbitrary software capability is already allowed

Canonical artifacts for this tranche:

- `fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_control_plane_decision_provenance_proof.rs`
- `scripts/check-tassadar-post-article-control-plane-decision-provenance-proof.sh`
