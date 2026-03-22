# Capability-Free Starter Plugin Scaffold

This directory was generated for:

- plugin id: `plugin.example.words`
- tool name: `plugin_example_words`
- authoring class: `capability_free_local_deterministic`

It is a bounded scaffold, not a finished plugin.

Generated files:

- `README.md`
- `scaffold_manifest.json`
- `plugin_example_words_runtime_snippet.rs`
- `tassadar_post_article_example_words_bundle.rs`
- `plugin_example_words_tests.rs`
- `check-example_words.sh`

How to use it:

1. copy the runtime snippet into
   `crates/psionic-runtime/src/tassadar_post_article_starter_plugin_runtime.rs`
2. replace every `TODO` with plugin-specific logic and negative claims
3. copy the example bundle writer into `crates/psionic-runtime/examples/`
4. merge the test stub into the runtime test module
5. copy the checker script into `scripts/`
6. decide explicitly whether the plugin should stay runtime-only or later be
   admitted to bridge, catalog, and controller surfaces

What it does not do:

- no automatic runtime-file patching
- no networked plugin scaffolding
- no automatic bridge, catalog, or controller admission
- no public publication or external binary support
