#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-sandbox --example tassadar_post_article_plugin_capability_boundary_report
cargo run -p psionic-research --example tassadar_post_article_plugin_capability_boundary_summary
cargo test -p psionic-provider post_article_plugin_capability_boundary_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_plugin_capability_boundary.report.v1"
  and .boundary_status == "green"
  and .boundary_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.reserved_capability_plane_id == "tassadar.post_article_universality_bridge.reserved_capability_plane.v1"
  and (.machine_identity_binding.canonical_architecture_anchor_crate == "psionic-transformer")
  and ((.dependency_rows | length) == 7)
  and ((.boundary_rows | length) == 7)
  and ((.state_receipt_rows | length) == 5)
  and ((.reserved_invariant_rows | length) == 3)
  and ((.validation_rows | length) == 10)
  and .tcm_v1_substrate_retained == true
  and .plugin_capability_plane_reserved == true
  and .plugin_execution_layer_separate == true
  and .downward_non_influence_reserved == true
  and .plugin_state_identity_separated == true
  and .closed_world_operator_curated_first_tranche_required == true
  and .first_plugin_tranche_posture == "closed_world_operator_curated_only_until_audited"
  and .choice_set_integrity_reserved == true
  and .resource_transparency_reserved == true
  and .scheduling_ownership_reserved == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == ["TAS-196", "TAS-197"])
' fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_capability_boundary.report.v1"
  and .boundary_status == "green"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .reserved_capability_plane_id == "tassadar.post_article_universality_bridge.reserved_capability_plane.v1"
  and .dependency_row_count == 7
  and .boundary_row_count == 7
  and .state_receipt_row_count == 5
  and .reserved_invariant_count == 3
  and .validation_row_count == 10
  and .first_plugin_tranche_posture == "closed_world_operator_curated_only_until_audited"
  and (.deferred_issue_ids == ["TAS-196", "TAS-197"])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_capability_boundary_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_rebased_universality_verdict_split.report.v1"
  and (.deferred_issue_ids == ["TAS-196"])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_split_report.json >/dev/null
