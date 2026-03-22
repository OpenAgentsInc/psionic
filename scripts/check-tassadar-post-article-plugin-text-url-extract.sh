#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_text_url_extract_bundle
cargo test -p psionic-runtime url_extract_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.plugin_text_url_extract.runtime_bundle.v1"
  and .plugin_id == "plugin.text.url_extract"
  and .plugin_version == "v1"
  and .packet_abi_version == "packet.v1"
  and .mount_envelope_id == "mount.plugin.text.url_extract.no_capabilities.v1"
  and .tool_projection.tool_name == "plugin_text_url_extract"
  and .tool_projection.replay_class_id == "deterministic_replayable"
  and ((.tool_projection.refusal_schema_ids | length) == 4)
  and ((.negative_claim_ids | length) == 4)
  and ((.case_rows | length) == 5)
  and (.case_rows | any(.case_id == "extract_urls_success" and .status == "exact_success"))
  and (.case_rows | any(.case_id == "schema_invalid_missing_text" and .response_or_refusal_schema_id == "plugin.refusal.schema_invalid.v1"))
  and (.case_rows | any(.case_id == "packet_too_large_refusal" and .response_or_refusal_schema_id == "plugin.refusal.packet_too_large.v1"))
  and (.case_rows | any(.case_id == "unsupported_codec_refusal" and .response_or_refusal_schema_id == "plugin.refusal.unsupported_codec.v1"))
  and (.case_rows | any(.case_id == "runtime_resource_limit_refusal" and .response_or_refusal_schema_id == "plugin.refusal.runtime_resource_limit.v1"))
' fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json >/dev/null
