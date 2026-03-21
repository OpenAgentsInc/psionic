# TAS-190 Audit

`TAS-190` closes the proof-rebinding tranche for the post-`TAS-186`
universality bridge.

The canonical artifact is now
`fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_report.json`.
It rebinds the historical universal-machine proof onto the declared bridge
machine identity, canonical model artifact, canonical weight artifact, and
canonical route id through one explicit proof-transport boundary instead of
treating rebinding as metadata relabeling.

The report keeps the preserved transition classes explicit, names the admitted
identity-binding variance, and blocks helper substitution, route-family drift,
undeclared cache-owned control, undeclared batching semantics, and semantic
drift outside the declared proof boundary.

The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_universal_machine_proof_rebinding.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-universal-machine-proof-rebinding.sh`.

This tranche is intentionally narrower than a full rebased universality
approval. It does not yet reissue the broader witness suite on the canonical
route, enable the canonical-route universal-substrate gate, publish the
rebased theory/operator/served verdict split, admit served/public
universality, admit weighted plugin control, or admit arbitrary software
capability.
