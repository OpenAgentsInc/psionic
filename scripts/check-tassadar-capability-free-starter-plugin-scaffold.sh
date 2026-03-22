#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FIXTURE_DIR="fixtures/tassadar/scaffolds/starter_plugin_capability_free_scaffold_v1"
GENERATOR="scripts/scaffold-tassadar-capability-free-starter-plugin.py"

tempdir="$(mktemp -d)"
trap 'rm -rf "$tempdir"' EXIT

python3 "$GENERATOR" \
  --plugin-id "plugin.example.words" \
  --output-dir "$tempdir"

diff -ru "$FIXTURE_DIR" "$tempdir"

jq -e '
  .schema_version == 1
  and .plugin_id == "plugin.example.words"
  and .tool_name == "plugin_example_words"
  and .authoring_class == "capability_free_local_deterministic"
  and .bridge_exposed_default == false
  and .catalog_exposed_default == false
' "$FIXTURE_DIR/scaffold_manifest.json" >/dev/null

rg -n "bridge_exposed: false" "$FIXTURE_DIR/plugin_example_words_runtime_snippet.rs" >/dev/null
rg -n "catalog_exposed: false" "$FIXTURE_DIR/plugin_example_words_runtime_snippet.rs" >/dev/null
rg -n 'todo!\("implement invoke_example_words_json_packet"\)' "$FIXTURE_DIR/plugin_example_words_runtime_snippet.rs" >/dev/null
rg -n "cargo run -p psionic-runtime --example tassadar_post_article_example_words_bundle" "$FIXTURE_DIR/check-example_words.sh" >/dev/null
