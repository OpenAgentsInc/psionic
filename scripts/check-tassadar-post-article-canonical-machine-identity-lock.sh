#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_canonical_machine_identity_lock_report
cargo run -p psionic-research --example tassadar_post_article_canonical_machine_identity_lock_summary

cargo test -p psionic-transformer canonical_machine_identity_lock_contract -- --nocapture
cargo test -p psionic-eval canonical_machine_identity_lock_report_ -- --nocapture
cargo test -p psionic-research canonical_machine_identity_lock_summary_ -- --nocapture
cargo test -p psionic-provider post_article_canonical_machine_identity_lock_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_canonical_machine_identity_lock.report.v1"
  and .lock_status == "green"
  and .lock_green == true
  and .canonical_machine_tuple.tuple_id == "tassadar.post_article.canonical_machine_identity_lock.tuple.v1"
  and .canonical_machine_tuple.carrier_class_id == "tassadar.post_article.canonical_machine.closure_bundle_bound_rebased_route_identity.v1"
  and .canonical_machine_tuple.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_machine_tuple.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_machine_tuple.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .canonical_machine_tuple.continuation_contract_id == "tassadar.tcm_v1.runtime_contract.report.v1"
  and ((.supporting_material_rows | length) == 8)
  and ((.dependency_rows | length) == 6)
  and ((.artifact_binding_rows | length) == 16)
  and ((.invalidation_rows | length) == 7)
  and ((.validation_rows | length) == 8)
  and .one_canonical_machine_named == true
  and .mixed_carrier_evidence_bundle_refused == true
  and .legacy_projection_binding_complete == true
  and .closure_bundle_embedded_here == false
  and .closure_bundle_issue_id == "TAS-215"
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == true
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (([.invalidation_rows[] | select(.present == true)] | length) == 0)
' fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_machine_identity_lock.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .carrier_class_id == "tassadar.post_article.canonical_machine.closure_bundle_bound_rebased_route_identity.v1"
  and .canonical_machine_lock_contract_id == "tassadar.post_article.canonical_machine_identity_lock.contract.v1"
  and .lock_status == "green"
  and .supporting_material_row_count == 8
  and .dependency_row_count == 6
  and .artifact_binding_row_count == 16
  and .invalidation_row_count == 7
  and .validation_row_count == 8
  and .one_canonical_machine_named == true
  and .mixed_carrier_evidence_bundle_refused == true
  and .legacy_projection_binding_complete == true
  and .closure_bundle_embedded_here == false
  and .closure_bundle_issue_id == "TAS-215"
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == true
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_summary.json >/dev/null

jq -e '
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and (.reserved_capability_issue_ids | index("TAS-195")) != null
  and (.reserved_capability_issue_ids | index("TAS-211")) != null
  and (.reserved_capability_issue_ids | index("TAS-207")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
