#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_plugin_result_binding_schema_stability_and_composition_report
cargo run -p psionic-research --example tassadar_post_article_plugin_result_binding_schema_stability_and_composition_summary
cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-runtime --example tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_report
cargo run -p psionic-eval --example tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_eval_report
cargo run -p psionic-research --example tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_summary
cargo test -p psionic-provider post_article_plugin_result_binding_receipt_projects_summary -- --nocapture
cargo test -p psionic-provider post_article_universality_bridge_contract -- --nocapture
cargo test -p psionic-provider post_article_weighted_plugin_controller_receipt_projects_summary -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop.runtime_bundle.v1"
  and .control_trace_contract_id == "tassadar.weighted_plugin.controller_trace_contract.v1"
  and .control_trace_profile_id == "tassadar.weighted_plugin.control_trace_profile.v1"
  and .determinism_profile_id == "tassadar.weighted_plugin.controller_determinism_profile.v1"
  and ((.controller_case_rows | length) == 4)
  and ((.control_trace_rows | length) == 34)
  and ((.host_negative_rows | length) == 10)
' fixtures/tassadar/runs/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_v1/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .plugin_selection_model_owned == true
  and .export_selection_model_owned == true
  and .packet_arguments_model_owned == true
  and .multi_step_sequencing_model_owned == true
  and .retry_decisions_model_owned == true
  and .stop_conditions_model_owned == true
  and .typed_refusal_returned_to_model_loop == true
  and .host_executes_but_is_not_planner == true
  and .determinism_contract_explicit == true
  and .external_signal_boundary_closed == true
  and .hidden_host_orchestration_negative_rows_green == true
  and .adversarial_host_behavior_negative_rows_green == true
  and .weighted_plugin_control_allowed == true
  and .plugin_capability_claim_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop.eval_report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .control_trace_contract_green == true
  and .determinism_profile_explicit == true
  and .typed_refusal_loop_closed == true
  and .host_not_planner_green == true
  and .adversarial_negative_rows_green == true
  and .weighted_plugin_control_allowed == true
  and .plugin_capability_claim_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_eval_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop.eval_report.v1"
  and .control_trace_contract_id == "tassadar.weighted_plugin.controller_trace_contract.v1"
  and .control_trace_profile_id == "tassadar.weighted_plugin.control_trace_profile.v1"
  and .determinism_profile_id == "tassadar.weighted_plugin.controller_determinism_profile.v1"
  and .contract_status == "green"
  and .dependency_row_count == 4
  and .controller_case_row_count == 4
  and .control_trace_row_count == 34
  and .host_negative_row_count == 10
  and .validation_row_count == 9
  and .control_trace_contract_green == true
  and .determinism_profile_explicit == true
  and .typed_refusal_loop_closed == true
  and .host_not_planner_green == true
  and .adversarial_negative_rows_green == true
  and .weighted_plugin_control_allowed == true
  and .plugin_capability_claim_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_result_binding_schema_stability_and_composition.report.v1"
  and (.deferred_issue_ids == [])
  and .weighted_plugin_control_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_result_binding_schema_stability_and_composition.report.v1"
  and (.deferred_issue_ids == [])
  and .weighted_plugin_control_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_result_binding_schema_stability_and_composition_summary.json >/dev/null

jq -e '
  (.reserved_capability_issue_ids | index("TAS-195")) != null
  and (.reserved_capability_issue_ids | index("TAS-211")) != null
  and (.reserved_capability_issue_ids | index("TAS-206")) == null
  and (.reserved_capability_issue_ids | index("TAS-207")) == null
  and (.reserved_capability_issue_ids | index("TAS-204")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
