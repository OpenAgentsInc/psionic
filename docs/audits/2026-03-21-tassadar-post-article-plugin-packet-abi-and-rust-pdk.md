# TAS-199 Audit

`TAS-199` closes the canonical plugin packet ABI and Rust-first PDK tranche
above the rebased post-`TAS-186` machine.

The canonical artifact is now
`fixtures/tassadar/reports/tassadar_post_article_plugin_packet_abi_and_rust_pdk_report.json`.
It binds the packet ABI and guest-authoring contract to the same canonical
bridge machine identity, canonical route, continuation contract, and
computational-model statement as the manifest contract, and it inherits the
closed manifest artifact plus the earlier internal component-ABI report as
explicit machine-checkable dependencies.

The report freezes one `packet.v1` invocation contract:

- one input packet
- one output packet or typed refusal
- one explicit host-error channel
- one explicit host receipt channel
- explicit schema ids, codec ids, payload bytes, and metadata envelopes

It also freezes one Rust-first guest surface:

- one Rust crate path
- one `handle_packet` export
- one `PluginRefusalV1` family
- one narrow packet-host import namespace
- no ambient authority and no hidden side-channel data

The matching runtime bundle now lives at
`fixtures/tassadar/runs/tassadar_post_article_plugin_packet_abi_and_rust_pdk_v1/tassadar_post_article_plugin_packet_abi_and_rust_pdk_bundle.json`.
That bundle keeps exact output-packet, typed-refusal, and host-error cases
machine-readable so the ABI stays auditable and challengeable instead of
being described only in prose.

The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_plugin_packet_abi_and_rust_pdk_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_plugin_packet_abi_and_rust_pdk.rs`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-plugin-packet-abi-and-rust-pdk.sh`.

This tranche is necessary for later host-owned runtime API, engine
abstraction, controller-trace, and publication/trust work, but it is still
not the final weighted-plugin control closure, public plugin platform,
served/public universality, or arbitrary software capability closeout.
