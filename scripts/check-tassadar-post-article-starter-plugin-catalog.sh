#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-catalog --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report
cargo run -p psionic-research --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary
cargo run -p psionic-eval --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_audit_report
cargo run -p psionic-research --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_summary
cargo run -p psionic-runtime --example tassadar_post_article_starter_plugin_catalog_bundle
cargo run -p psionic-catalog --example tassadar_post_article_starter_plugin_catalog_report
cargo run -p psionic-eval --example tassadar_post_article_starter_plugin_catalog_eval_report
cargo run -p psionic-research --example tassadar_post_article_starter_plugin_catalog_summary

cargo test -p psionic-runtime starter_plugin_catalog_ -- --nocapture
cargo test -p psionic-catalog starter_plugin_catalog_ -- --nocapture
cargo test -p psionic-eval starter_plugin_catalog_ -- --nocapture
cargo test -p psionic-research starter_plugin_catalog_ -- --nocapture
cargo test -p psionic-provider post_article_starter_plugin_catalog_receipt_projects_summary -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.starter_plugin_catalog.runtime_bundle.v1"
  and .plugin_count == 6
  and .local_deterministic_plugin_count == 5
  and .read_only_network_plugin_count == 1
  and .bounded_flow_count == 2
  and ((.descriptor_rows | length) == 6)
  and ((.capability_matrix_rows | length) == 6)
  and ((.composition_case_rows | length) == 2)
  and .operator_only_posture == true
  and .runtime_builtins_separate == true
  and .public_marketplace_implication_allowed == false
  and (.descriptor_rows | any(.plugin_id == "plugin.text.url_extract"))
  and (.descriptor_rows | any(.plugin_id == "plugin.text.stats"))
  and (.descriptor_rows | any(.plugin_id == "plugin.http.fetch_text"))
  and (.descriptor_rows | any(.plugin_id == "plugin.example.echo_guest"))
  and (.descriptor_rows | any(.plugin_id == "plugin.html.extract_readable"))
  and (.descriptor_rows | any(.plugin_id == "plugin.feed.rss_atom_parse"))
' fixtures/tassadar/runs/tassadar_post_article_starter_plugin_catalog_v1/tassadar_post_article_starter_plugin_catalog_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article.starter_plugin_catalog.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and ((.dependency_rows | length) == 7)
  and ((.entry_rows | length) == 6)
  and ((.capability_rows | length) == 6)
  and ((.composition_rows | length) == 2)
  and ((.validation_rows | length) == 8)
  and .operator_internal_only_posture == true
  and .local_network_distinction_explicit == true
  and .descriptor_refs_complete == true
  and .fixture_bundle_refs_complete == true
  and .sample_mount_envelope_refs_complete == true
  and .composition_harness_green == true
  and .runtime_builtins_separate == true
  and .public_marketplace_language_suppressed == true
  and .closure_bundle_bound_by_digest == true
  and .plugin_capability_claim_allowed == true
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .next_issue_id == "TAS-217"
' fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article.starter_plugin_catalog.eval_report.v1"
  and .eval_status == "green"
  and .eval_green == true
  and ((.dependency_rows | length) == 2)
  and ((.validation_rows | length) == 4)
  and .starter_plugin_count == 6
  and .local_deterministic_plugin_count == 5
  and .read_only_network_plugin_count == 1
  and .bounded_flow_count == 2
  and .operator_internal_only_posture == true
  and .public_marketplace_language_suppressed == true
  and .closure_bundle_bound_by_digest == true
  and .plugin_capability_claim_allowed == true
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .next_issue_id == "TAS-217"
' fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_eval_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article.starter_plugin_catalog.eval_report.v1"
  and .eval_status == "green"
  and .starter_plugin_count == 6
  and .local_deterministic_plugin_count == 5
  and .read_only_network_plugin_count == 1
  and .bounded_flow_count == 2
  and .operator_internal_only_posture == true
  and .public_marketplace_language_suppressed == true
  and .closure_bundle_bound_by_digest == true
  and .plugin_capability_claim_allowed == true
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .next_issue_id == "TAS-217"
' fixtures/tassadar/reports/tassadar_post_article_starter_plugin_catalog_summary.json >/dev/null
