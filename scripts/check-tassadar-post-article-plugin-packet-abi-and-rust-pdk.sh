#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_packet_abi_and_rust_pdk_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_packet_abi_and_rust_pdk_report
cargo run -p psionic-research --example tassadar_post_article_plugin_packet_abi_and_rust_pdk_summary
cargo test -p psionic-provider post_article_plugin_packet_abi_receipt_projects_summary -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article_plugin_packet_abi_and_rust_pdk.runtime_bundle.v1"
  and .packet_abi_version == "packet.v1"
  and .rust_first_pdk_id == "tassadar.plugin.rust_first_pdk.v1"
  and .host_import_namespace_id == "tassadar.plugin_host.packet.v1"
  and ((.packet_field_ids | length) == 4)
  and ((.host_imports | length) == 3)
  and ((.typed_refusals | length) == 2)
  and (.host_error_channel_ids == ["capability_namespace_unmounted"])
  and ((.receipt_field_ids | length) == 5)
  and ((.case_receipts | length) == 5)
' fixtures/tassadar/runs/tassadar_post_article_plugin_packet_abi_and_rust_pdk_v1/tassadar_post_article_plugin_packet_abi_and_rust_pdk_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_packet_abi_and_rust_pdk.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .machine_identity_binding.packet_abi_version == "packet.v1"
  and .machine_identity_binding.rust_first_pdk_id == "tassadar.plugin.rust_first_pdk.v1"
  and ((.dependency_rows | length) == 4)
  and ((.abi_rows | length) == 8)
  and ((.pdk_rows | length) == 6)
  and ((.validation_rows | length) == 8)
  and .operator_internal_only_posture == true
  and .packet_abi_frozen == true
  and .rust_first_pdk_frozen == true
  and .typed_refusal_channel_frozen == true
  and .explicit_host_error_channel_frozen == true
  and .explicit_receipt_channel_required == true
  and .narrow_host_import_surface_frozen == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == ["TAS-200"])
' fixtures/tassadar/reports/tassadar_post_article_plugin_packet_abi_and_rust_pdk_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_packet_abi_and_rust_pdk.report.v1"
  and .packet_abi_version == "packet.v1"
  and .rust_first_pdk_id == "tassadar.plugin.rust_first_pdk.v1"
  and .contract_status == "green"
  and .dependency_row_count == 4
  and .abi_row_count == 8
  and .pdk_row_count == 6
  and .validation_row_count == 8
  and (.deferred_issue_ids == ["TAS-200"])
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_packet_abi_and_rust_pdk_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_manifest_identity_contract.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_report.json >/dev/null
