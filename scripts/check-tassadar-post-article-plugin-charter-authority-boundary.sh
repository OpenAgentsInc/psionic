#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-sandbox --example tassadar_post_article_plugin_charter_authority_boundary_report
cargo run -p psionic-research --example tassadar_post_article_plugin_charter_authority_boundary_summary
cargo test -p psionic-provider post_article_plugin_charter_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_plugin_charter_authority_boundary.report.v1"
  and .charter_status == "green"
  and .charter_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and ((.dependency_rows | length) == 11)
  and ((.law_rows | length) == 17)
  and ((.state_class_rows | length) == 4)
  and ((.governance_rows | length) == 4)
  and ((.validation_rows | length) == 10)
  and .current_publication_posture == "internal_only_until_later_plugin_platform_gates"
  and .first_plugin_tranche_posture == "closed_world_operator_curated_only_until_audited"
  and .internal_only_plugin_posture == true
  and .proof_vs_audit_distinction_frozen == true
  and .observer_model_frozen == true
  and .three_plane_contract_frozen == true
  and .adversarial_host_model_frozen == true
  and .state_class_split_frozen == true
  and .host_executes_but_does_not_decide == true
  and .semantic_preservation_required == true
  and .choice_set_integrity_frozen == true
  and .resource_transparency_frozen == true
  and .scheduling_ownership_frozen == true
  and .no_externalized_learning_frozen == true
  and .plugin_language_boundary_frozen == true
  and .first_plugin_tranche_closed_world == true
  and .anti_interpreter_smuggling_frozen == true
  and .downward_non_influence_frozen == true
  and .governance_receipts_required == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == ["TAS-198"])
' fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_charter_authority_boundary.report.v1"
  and .charter_status == "green"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .dependency_row_count == 11
  and .law_row_count == 17
  and .state_class_row_count == 4
  and .governance_row_count == 4
  and .validation_row_count == 10
  and .current_publication_posture == "internal_only_until_later_plugin_platform_gates"
  and .first_plugin_tranche_posture == "closed_world_operator_curated_only_until_audited"
  and (.deferred_issue_ids == ["TAS-198"])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_capability_boundary.report.v1"
  and (.deferred_issue_ids == [])
  and .plugin_capability_claim_allowed == false
  and .plugin_publication_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_report.json >/dev/null
