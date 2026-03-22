# Tassadar Starter Plugin Authoring

This document is the bounded authoring contract for adding a new starter plugin
inside this repository.

The first-class path in this pass is intentionally narrow:

- host-native Rust implementation only
- capability-free local deterministic plugin only
- explicit packet schemas, typed refusals, receipts, and runtime artifacts
- no public marketplace, external binary loading, or automatic publication

One manual `networked_read_only` proof now exists through
`plugin.http.fetch_text`, but it remains a narrower hand-built path with
explicit mount and replay policy rather than the low-friction default.

If a contributor can follow this document, they should be able to add a new
capability-free starter plugin without reverse engineering unrelated files.

## Authoring Classes

Two starter-plugin authoring classes now matter explicitly:

### `capability_free_local_deterministic`

- no capability namespaces
- no network mount envelope configuration
- deterministic replay posture
- standard capability-free refusal and negative-claim pattern
- scaffold helper supported

### `networked_read_only`

- explicit capability namespace and mount envelope required
- replay posture must stay explicit, usually snapshot-backed or operator-only
- network-policy, timeout, redirect, and response-size truth must stay
  machine-legible
- no scaffold helper in this first pass

The low-risk class is intentionally easier. The networked class keeps its extra
policy burden explicit instead of inheriting unsafe defaults from the
capability-free path.

## Naming Contract

Choose one stable naming family and keep it consistent everywhere:

- plugin id:
  `plugin.<domain>.<name>`
- tool name:
  `plugin_<domain>_<name>`
- input schema id:
  `plugin.<domain>.<name>.input.v1`
- success output schema id:
  `plugin.<domain>.<name>.output.v1`
- mount envelope id:
  `mount.plugin.<domain>.<name>.no_capabilities.v1`
- runtime bundle id:
  `tassadar.post_article.plugin_<domain>_<name>.runtime_bundle.v1`

For a capability-free starter plugin, keep:

- `replay_class_id = deterministic_replayable`
- `capability_class = LocalDeterministic`
- `capability_namespace_ids = []`

## Required Runtime Pieces

Every capability-free starter plugin needs the same bounded runtime pieces in
`crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`:

1. request and response structs
2. config struct, if the plugin has bounded runtime limits
3. invocation-outcome struct
4. runtime-case enum and runtime-case row type
5. runtime-bundle type
6. refusal-schema list and negative-claim list
7. one `StarterPluginRegistration` row in the central registry
8. one tool-projection function
9. one invocation function that:
   - decodes a typed packet
   - fails closed into typed refusals
   - returns a success receipt on success
10. one runtime-bundle builder plus write/load helpers

The runtime is the source of truth. Bridge, catalog, and later controller
surfaces derive from that registration instead of inventing parallel metadata.

The registry now also carries a machine-readable `origin_class`. Use
`operator_builtin` for the original repo-owned starter set and `user_added`
for new contributor-authored starter plugins such as `plugin.text.stats` and
the first manual `networked_read_only` proof target `plugin.http.fetch_text`.

## Minimum Registration Template

Use the central registry row as the canonical contract:

```rust
StarterPluginRegistration {
    plugin_id: "plugin.example.words",
    plugin_version: STARTER_PLUGIN_VERSION,
    tool_name: "plugin_example_words",
    input_schema_id: "plugin.example.words.input.v1",
    success_output_schema_id: "plugin.example.words.output.v1",
    refusal_schema_ids: EXAMPLE_REFUSAL_SCHEMA_IDS,
    replay_class_id: "deterministic_replayable",
    capability_class: StarterPluginCapabilityClass::LocalDeterministic,
    capability_namespace_ids: NO_CAPABILITY_NAMESPACE_IDS,
    origin_class: StarterPluginOriginClass::UserAdded,
    negative_claim_ids: EXAMPLE_NEGATIVE_CLAIM_IDS,
    mount_envelope_id: "mount.plugin.example.words.no_capabilities.v1",
    manifest_id: "manifest.plugin.example.words.v1",
    artifact_id: "artifact.plugin.example.words.v1",
    runtime_bundle_id: "tassadar.post_article.plugin_example_words.runtime_bundle.v1",
    runtime_bundle_ref: "...",
    runtime_run_root_ref: "...",
    tool_description: "...",
    bridge_exposed: false,
    catalog_exposed: false,
    catalog: None,
}
```

Start with `bridge_exposed = false` and `catalog_exposed = false` if the new
plugin is runtime-only. Later issues or explicit follow-on work can widen it to
the shared bridge, catalog, and controller lanes.

For the bounded canonical weighted-controller lane, the current admission rule
is also explicit: only `user_added` starter plugins that are either
`capability_free_local_deterministic` or `networked_read_only`, are
bridge-exposed, and are catalog-exposed are eligible for the shared admission
helper consumed by the weighted-controller trace.

## Refusal Contract

Capability-free starter plugins should prefer the standard bounded refusal set:

- `plugin.refusal.schema_invalid.v1`
- `plugin.refusal.packet_too_large.v1`
- `plugin.refusal.unsupported_codec.v1`

Add extra refusal ids only when the plugin truly needs them. Keep the reason
typed and machine-readable. Do not return free-form failure text in place of a
typed refusal packet.

## Artifact Contract

At minimum, land these task-owned artifact surfaces:

- runtime bundle under `fixtures/tassadar/runs/...`
- one example writer in `crates/psionic-runtime/examples/`
- one targeted checker in `scripts/`
- targeted runtime tests
- one runtime doc update

If the plugin is admitted to later surfaces, also land:

- shared bridge artifact updates
- starter catalog sidecars and downstream reports
- controller artifacts only when a controller issue explicitly widens there

## Contributor Checklist

For a new capability-free plugin, the normal sequence is:

1. implement the runtime structs, invocation path, and bundle builder
2. add the central registration row
3. add targeted runtime tests
4. add the example bundle writer
5. add the checker script
6. update the runtime doc
7. run the targeted validation for the new plugin
8. only then decide whether to admit it to bridge, catalog, or controller
   surfaces

## Scaffold Helper

The repo now also ships one narrow scaffold helper for the
`capability_free_local_deterministic` class:

```bash
python3 scripts/scaffold-tassadar-capability-free-starter-plugin.py \
  --plugin-id plugin.example.words \
  --authoring-class capability_free_local_deterministic \
  --output-dir /tmp/plugin-example-words
```

The scaffold writes a bounded stub tree:

- runtime snippet
- bundle-writer stub
- test stub
- checker stub
- one manifest describing the generated ids and defaults

It does not patch live runtime files for you. The generated output is a
starting point that still requires explicit review and TODO completion.

If you pass `--authoring-class networked_read_only`, the scaffold refuses
generation on purpose and tells you to use the manual networked path instead.

## Example Path

`plugin.text.stats` is the canonical example for this authoring class:

- capability-free
- local deterministic
- typed input and output
- typed refusals
- explicit negative claims
- runtime bundle, checker, and demo example all committed

Use that plugin as the reference implementation when the checklist above feels
too abstract.

`plugin.http.fetch_text` is the canonical manual example for the
`networked_read_only` class:

- explicit capability namespace
- explicit mount envelope
- explicit replay posture
- typed network refusal surface
- shared bridge, catalog, workflow-controller, and weighted-controller proof

## Non-Goals

This authoring contract does not claim:

- a general external plugin SDK
- networked plugin authoring as the default path
- public plugin publication rights
- automatic controller or weighted-lane admission
- arbitrary binary loading
