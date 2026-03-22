#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_html_extract_readable_bundle
cargo test -p psionic-runtime extract_readable_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.plugin_html_extract_readable.runtime_bundle.v1"
  and .plugin_id == "plugin.html.extract_readable"
  and .plugin_version == "v1"
  and .packet_abi_version == "packet.v1"
  and .mount_envelope_id == "mount.plugin.html.extract_readable.no_capabilities.v1"
  and .tool_projection.tool_name == "plugin_html_extract_readable"
  and ((.negative_claim_ids | length) == 4)
  and ((.case_rows | length) == 5)
  and (.case_rows | any(.case_id == "extract_readable_success" and .status == "exact_success"))
  and (.case_rows | any(.case_id == "extract_readable_malformed_but_recoverable_success" and .status == "exact_success"))
  and (.case_rows | any(.case_id == "schema_invalid_missing_body_text" and .response_or_refusal_schema_id == "plugin.refusal.schema_invalid.v1"))
  and (.case_rows | any(.case_id == "content_type_unsupported_refusal" and .response_or_refusal_schema_id == "plugin.refusal.content_type_unsupported.v1"))
  and (.case_rows | any(.case_id == "input_too_large_refusal" and .response_or_refusal_schema_id == "plugin.refusal.input_too_large.v1"))
  and .composition_case.case_id == "fetch_then_extract_readable"
  and .composition_case.green == true
  and .composition_case.schema_repair_allowed == false
  and .composition_case.hidden_host_extraction_allowed == false
' fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1/tassadar_post_article_plugin_html_extract_readable_bundle.json >/dev/null
