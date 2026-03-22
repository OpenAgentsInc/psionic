#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# TODO: update the package/example names after copying this scaffold into the repo.
cargo run -p psionic-runtime --example tassadar_post_article_example_words_bundle
cargo test -p psionic-runtime example_words_ -- --nocapture

jq -e '
  .tool_projection.tool_name == "plugin_example_words"
  and .plugin_id != ""
  and .bundle_id != ""
' "TODO_REPLACE_WITH_RUNTIME_BUNDLE_PATH" >/dev/null
