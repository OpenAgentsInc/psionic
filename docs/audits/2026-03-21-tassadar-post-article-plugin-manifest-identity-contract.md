# TAS-198 Audit

`TAS-198` closes the canonical plugin manifest, identity, and hot-swap
contract tranche above the rebased post-`TAS-186` machine.

The canonical artifact is now
`fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_report.json`.
It binds the plugin-artifact contract to the same canonical bridge machine
identity, canonical route, continuation contract, and computational-model
statement as the plugin charter, and it inherits the green plugin charter,
module trust isolation, module promotion state, internal compute package
manager, and internal compute package route policy as explicit
machine-checkable dependencies.

The report freezes canonical plugin identity around `plugin_id`,
`plugin_version`, and `artifact_digest`, requires declared exports, packet ABI
version, input and output schema ids, limits, trust tier, replay class,
evidence settings, and publication posture, and makes canonical invocation
identity explicit by adding export name, packet ABI version, and
mount-envelope identity. It also freezes one typed hot-swap contract:
versioned replacement stays explicit, ABI and schema drift stay fail-closed,
trust widening requires promotion and trust receipts, and replay/evidence
posture remain part of compatibility instead of host-owned hidden state.

The report also keeps packaging explicit. Single-module plugins remain named
artifacts, linked multi-module bundles must name their member refs, and
future component-model bundles stay blocked until a later admitted profile
exists. Hidden host orchestration, schema drift, envelope leakage, side
channels, and overclaim posture remain validation rows rather than implied
assumptions.

The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_plugin_manifest_identity_contract.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-plugin-manifest-identity-contract.sh`.

This tranche is necessary for later packet-ABI, runtime API, receipt-family,
controller-trace, and publication/trust work, but it is still not the final
weighted-plugin control closure, public plugin platform, served/public
universality, or arbitrary software capability closeout.
