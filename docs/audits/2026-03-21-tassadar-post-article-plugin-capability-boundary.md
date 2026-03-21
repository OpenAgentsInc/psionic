# TAS-195 Audit

`TAS-195` closes the plugin-aware boundary tranche above the rebased
post-`TAS-186` `TCM.v1` substrate.

The canonical artifact is now
`fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_report.json`.
It binds the reserved plugin capability plane to the canonical bridge machine
identity, keeps `TCM.v1` as the bounded compute substrate, keeps plugin
execution on a separate software-capability layer, and states machine-readably
that plugin ergonomics may not rewrite continuation semantics, carrier
identity, or proof assumptions below that layer.

The report also freezes plugin packet and receipt identity as separate future
families, carries forward choice-set integrity, resource transparency, and
scheduling ownership as non-negotiable invariants, and reserves the first
audited plugin tranche as closed-world and operator-curated. It does this
without widening the rebased theory/operator verdict into weighted plugin
control, plugin publication, served/public universality, or arbitrary software
capability.

The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_plugin_capability_boundary.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-plugin-capability-boundary.sh`.

This tranche makes the rebased closeout explicitly plugin-aware. It is still
not the plugin charter, plugin manifest/ABI/runtime tranche, plugin
publication/trust gate, served/public universality, or arbitrary software
capability closeout.
