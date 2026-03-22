#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_canonical_computational_model_statement_report
cargo run -p psionic-eval --example tassadar_post_article_execution_semantics_proof_transport_audit_report
cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_canonical_machine_identity_lock_report
cargo run -p psionic-research --example tassadar_post_article_canonical_computational_model_statement_summary

cargo test -p psionic-transformer canonical_computational_model_contract -- --nocapture
cargo test -p psionic-runtime canonical_computational_model_statement_report_ -- --nocapture
cargo test -p psionic-research canonical_computational_model_statement_summary_ -- --nocapture
cargo test -p psionic-provider post_article_canonical_computational_model_statement_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_canonical_computational_model_statement.report.v1"
  and .statement_status == "green"
  and .statement_green == true
  and .computational_model_statement.statement_id == "tassadar.post_article.canonical_computational_model.statement.v1"
  and .computational_model_statement.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .computational_model_statement.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .computational_model_statement.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .computational_model_statement.runtime_contract_id == "tassadar.tcm_v1.runtime_contract.report.v1"
  and .computational_model_statement.substrate_model_id == "tcm.v1"
  and .proof_transport_audit_report_ref == "fixtures/tassadar/reports/tassadar_post_article_execution_semantics_proof_transport_audit_report.json"
  and ((.supporting_material_rows | length) == 11)
  and ((.dependency_rows | length) == 7)
  and ((.invalidation_rows | length) == 5)
  and ((.validation_rows | length) == 7)
  and .article_equivalent_compute_named == true
  and .tcm_v1_continuation_named == true
  and .declared_effect_boundary_named == true
  and .plugin_layer_scoped_above_machine == true
  and .proof_transport_complete == true
  and .proof_transport_audit_issue_id == "TAS-209"
  and .next_stability_issue_id == "TAS-211"
  and .closure_bundle_embedded_here == false
  and .closure_bundle_issue_id == "TAS-215"
  and .weighted_plugin_control_part_of_model == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (([.invalidation_rows[] | select(.present == true)] | length) == 0)
' fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_computational_model_statement.report.v1"
  and .statement_id == "tassadar.post_article.canonical_computational_model.statement.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .continuation_contract_id == "tassadar.tcm_v1.runtime_contract.report.v1"
  and .substrate_model_id == "tcm.v1"
  and .statement_status == "green"
  and .supporting_material_row_count == 11
  and .dependency_row_count == 7
  and .invalidation_row_count == 5
  and .validation_row_count == 7
  and .article_equivalent_compute_named == true
  and .tcm_v1_continuation_named == true
  and .declared_effect_boundary_named == true
  and .plugin_layer_scoped_above_machine == true
  and .proof_transport_complete == true
  and .proof_transport_audit_issue_id == "TAS-209"
  and .next_stability_issue_id == "TAS-211"
  and .closure_bundle_embedded_here == false
  and .closure_bundle_issue_id == "TAS-215"
  and .weighted_plugin_control_part_of_model == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_summary.json >/dev/null

jq -e '
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and (.reserved_capability_issue_ids | index("TAS-195")) != null
  and (.reserved_capability_issue_ids | index("TAS-211")) != null
  and (.reserved_capability_issue_ids | index("TAS-208")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
