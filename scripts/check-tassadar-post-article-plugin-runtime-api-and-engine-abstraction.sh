#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report
cargo run -p psionic-research --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary
cargo run -p psionic-serve --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication
cargo test -p psionic-provider post_article_plugin_runtime_api_receipt_projects_summary -- --nocapture
cargo test -p psionic-serve post_article_plugin_runtime_api_publication_keeps_served_surface_blocked -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.runtime_bundle.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and .engine_profile_id == "tassadar.plugin_runtime.engine_profile.wasm_bounded.v1"
  and .packet_abi_version == "packet.v1"
  and ((.artifact_loading_rows | length) == 3)
  and ((.engine_operation_rows | length) == 6)
  and ((.bound_rows | length) == 6)
  and ((.signal_rows | length) == 8)
  and ((.failure_isolation_rows | length) == 3)
  and ((.case_receipts | length) == 6)
  and .exact_success_case_count == 2
  and .exact_typed_refusal_case_count == 1
  and .exact_runtime_failure_case_count == 2
  and .exact_cancellation_case_count == 1
  and .time_semantics.logical_time_model_observable == false
  and .time_semantics.wall_time_model_observable == false
  and .time_semantics.queue_depth_model_observable == false
  and .time_semantics.retry_budget_model_observable == false
  and .time_semantics.runtime_cost_model_observable == false
  and .time_semantics.cost_model_invariance_required == true
  and .time_semantics.scheduling_semantics_fixed == true
' fixtures/tassadar/runs/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_v1/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .machine_identity_binding.packet_abi_version == "packet.v1"
  and .machine_identity_binding.host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .machine_identity_binding.engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and ((.dependency_rows | length) == 5)
  and ((.runtime_api_rows | length) == 8)
  and ((.engine_rows | length) == 6)
  and ((.bound_rows | length) == 6)
  and ((.signal_boundary_rows | length) == 8)
  and ((.failure_isolation_rows | length) == 3)
  and ((.validation_rows | length) == 8)
  and .operator_internal_only_posture == true
  and .runtime_api_frozen == true
  and .engine_abstraction_frozen == true
  and .artifact_digest_verification_required == true
  and .runtime_bounds_frozen == true
  and .model_information_boundary_frozen == true
  and .logical_time_control_neutral == true
  and .wall_time_control_neutral == true
  and .cost_model_invariance_required == true
  and .scheduling_semantics_frozen == true
  and .failure_domain_isolation_frozen == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.report.v1"
  and .packet_abi_version == "packet.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and .contract_status == "green"
  and .dependency_row_count == 5
  and .runtime_api_row_count == 8
  and .engine_row_count == 6
  and .bound_row_count == 6
  and .signal_boundary_row_count == 8
  and .failure_isolation_row_count == 3
  and .validation_row_count == 8
  and (.deferred_issue_ids == [])
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary.json >/dev/null

jq -e '
  .publication_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.publication.v1"
  and .current_served_internal_compute_profile_id == "tassadar.internal_compute.article_closeout.v1"
  and .packet_abi_version == "packet.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and (.served_plugin_surface_ids == [])
  and (.blocked_by == [])
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_packet_abi_and_rust_pdk.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_packet_abi_and_rust_pdk_report.json >/dev/null
