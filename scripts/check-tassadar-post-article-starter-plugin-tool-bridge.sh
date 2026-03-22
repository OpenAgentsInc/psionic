#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_starter_plugin_tool_bridge_bundle
cargo test -p psionic-runtime starter_plugin_tool_bridge_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.starter_plugin_tool_bridge.bundle.v1"
  and ((.surface_ids | length) == 3)
  and ((.projection_rows | length) == 4)
  and ((.execution_cases | length) == 4)
  and (.projection_rows | all(.stable_across_surfaces == true))
  and (.projection_rows | any(.tool_name == "plugin_text_url_extract"))
  and (.projection_rows | any(.tool_name == "plugin_http_fetch_text"))
  and (.projection_rows | any(.tool_name == "plugin_html_extract_readable"))
  and (.projection_rows | any(.tool_name == "plugin_feed_rss_atom_parse"))
  and (.execution_cases | any(.case_id == "feed_parse_refusal_bridge" and .status == "refusal" and .typed_refusal_preserved == true))
  and (.execution_cases | all(.receipt_binding_preserved == true))
' fixtures/tassadar/runs/tassadar_post_article_starter_plugin_tool_bridge_v1/tassadar_post_article_starter_plugin_tool_bridge_bundle.json >/dev/null
