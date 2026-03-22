#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_http_fetch_text_bundle
cargo test -p psionic-runtime fetch_text_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.plugin_http_fetch_text.runtime_bundle.v1"
  and .plugin_id == "plugin.http.fetch_text"
  and .plugin_version == "v1"
  and .packet_abi_version == "packet.v1"
  and .sample_mount_envelope.envelope_id == "mount.plugin.http.fetch_text.read_only_http_allowlist.v1"
  and .tool_projection.tool_name == "plugin_http_fetch_text"
  and ((.supported_replay_class_ids | length) == 2)
  and ((.negative_claim_ids | length) == 5)
  and ((.case_rows | length) == 9)
  and (.case_rows | any(.case_id == "fetch_text_article_success" and .status == "exact_success"))
  and (.case_rows | any(.case_id == "schema_invalid_missing_url" and .response_or_refusal_schema_id == "plugin.refusal.schema_invalid.v1"))
  and (.case_rows | any(.case_id == "url_not_permitted_refusal" and .response_or_refusal_schema_id == "plugin.refusal.url_not_permitted.v1"))
  and (.case_rows | any(.case_id == "timeout_refusal" and .response_or_refusal_schema_id == "plugin.refusal.timeout.v1"))
  and (.case_rows | any(.case_id == "network_denied_refusal" and .response_or_refusal_schema_id == "plugin.refusal.network_denied.v1"))
  and (.case_rows | any(.case_id == "response_too_large_refusal" and .response_or_refusal_schema_id == "plugin.refusal.response_too_large.v1"))
  and (.case_rows | any(.case_id == "content_type_unsupported_refusal" and .response_or_refusal_schema_id == "plugin.refusal.content_type_unsupported.v1"))
  and (.case_rows | any(.case_id == "decode_failed_refusal" and .response_or_refusal_schema_id == "plugin.refusal.decode_failed.v1"))
  and (.case_rows | any(.case_id == "upstream_failure_refusal" and .response_or_refusal_schema_id == "plugin.refusal.upstream_failure.v1"))
' fixtures/tassadar/runs/tassadar_post_article_plugin_http_fetch_text_v1/tassadar_post_article_plugin_http_fetch_text_bundle.json >/dev/null
