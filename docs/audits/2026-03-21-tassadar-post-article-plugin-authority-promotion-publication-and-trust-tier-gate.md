# TAS-205 Plugin Authority, Promotion, Publication, And Trust-Tier Gate

## Scope

This audit note records the first machine-readable gate above the canonical
weighted-plugin controller that freezes bounded plugin authority, promotion,
publication posture, and trust tiers without widening the claim surface into a
served/public plugin platform or arbitrary public software execution claim.

## Canonical Artifacts

- `fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary.json`
- `crates/psionic-provider/src/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate.rs`
- `scripts/check-tassadar-post-article-plugin-authority-promotion-publication-and-trust-tier-gate.sh`

## Boundary

The gate inherits the canonical post-`TAS-186` machine identity, the
weighted-plugin controller proof, and the manifest identity contract, then
binds them to explicit module trust tiers, promotion lifecycle receipts,
profile-specific publication posture, observer rights, and validator plus
accepted-outcome hook requirements. The current posture remains
operator/internal only, keeps profile-specific deterministic-import and
runtime-support lanes explicitly suppressed behind named policy hooks, refuses
broader public publication, and now serves as one prerequisite beneath the
separate `TAS-206` bounded plugin-platform closeout audit.
