#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_canonical_computational_model_statement_report
cargo run -p psionic-research --example tassadar_post_article_canonical_computational_model_statement_summary
cargo run -p psionic-eval --example tassadar_post_article_execution_semantics_proof_transport_audit_report
cargo run -p psionic-research --example tassadar_post_article_execution_semantics_proof_transport_audit_summary
cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-eval --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report
cargo run -p psionic-research --example tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_summary
cargo run -p psionic-catalog --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report
cargo run -p psionic-research --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary
cargo run -p psionic-eval --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_audit_report
cargo run -p psionic-research --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_summary
cargo run -p psionic-eval --example tassadar_post_article_canonical_machine_identity_lock_report
cargo run -p psionic-research --example tassadar_post_article_canonical_machine_identity_lock_summary
cargo run -p psionic-eval --example tassadar_post_article_continuation_non_computationality_contract_report
cargo run -p psionic-research --example tassadar_post_article_continuation_non_computationality_contract_summary

cargo test -p psionic-transformer continuation_non_computationality_contract -- --nocapture
cargo test -p psionic-eval continuation_non_computationality_contract_report_ -- --nocapture
cargo test -p psionic-research continuation_non_computationality_contract_summary_ -- --nocapture
cargo test -p psionic-provider post_article_continuation_non_computationality_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_continuation_non_computationality_contract.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .continuation_non_computationality_complete == true
  and .continuation_extends_execution_without_second_machine == true
  and .hidden_workflow_logic_refused == true
  and .continuation_expressivity_extension_blocked == true
  and .plugin_resume_hidden_compute_refused == true
  and .machine_identity_binding.tuple_id == "tassadar.post_article.canonical_machine_identity_lock.tuple.v1"
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.continuation_contract_id == "tassadar.tcm_v1.runtime_contract.report.v1"
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article.canonical_computational_model.statement.v1"
  and .machine_identity_binding.proof_transport_boundary_id == "tassadar.post_article_universal_machine_proof.proof_transport_boundary.v1"
  and ((.supporting_material_rows | length) == 10)
  and ((.dependency_rows | length) == 6)
  and ((.continuation_surface_rows | length) == 6)
  and ((.invalidation_rows | length) == 7)
  and ((.validation_rows | length) == 9)
  and (([.invalidation_rows[] | select(.present == true)] | length) == 0)
  and (([.validation_rows[] | select(.green == false)] | length) == 0)
  and (.continuation_surface_rows | all(.surface_green == true))
  and .next_stability_issue_id == "TAS-211"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_continuation_non_computationality_contract.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .continuation_contract_id == "tassadar.tcm_v1.runtime_contract.report.v1"
  and .computational_model_statement_id == "tassadar.post_article.canonical_computational_model.statement.v1"
  and .proof_transport_boundary_id == "tassadar.post_article_universal_machine_proof.proof_transport_boundary.v1"
  and .contract_status == "green"
  and .supporting_material_row_count == 10
  and .dependency_row_count == 6
  and .continuation_surface_row_count == 6
  and .invalidation_row_count == 7
  and .validation_row_count == 9
  and .continuation_extends_execution_without_second_machine == true
  and .hidden_workflow_logic_refused == true
  and .continuation_expressivity_extension_blocked == true
  and .plugin_resume_hidden_compute_refused == true
  and .continuation_non_computationality_complete == true
  and .next_stability_issue_id == "TAS-211"
  and .closure_bundle_issue_id == "TAS-215"
' fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_execution_semantics_proof_transport_audit.report.v1"
  and .next_stability_issue_id == "TAS-211"
  and .audit_status == "green"
  and .audit_green == true
' fixtures/tassadar/reports/tassadar_post_article_execution_semantics_proof_transport_audit_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_computational_model_statement.report.v1"
  and .next_stability_issue_id == "TAS-211"
  and .statement_status == "green"
  and .statement_green == true
' fixtures/tassadar/reports/tassadar_post_article_canonical_computational_model_statement_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_canonical_machine_identity_lock.report.v1"
  and .lock_status == "green"
  and .lock_green == true
' fixtures/tassadar/reports/tassadar_post_article_canonical_machine_identity_lock_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_bounded_weighted_plugin_platform_closeout_audit.report.v1"
  and .closeout_status == "operator_green_served_suppressed"
  and .closeout_green == true
' fixtures/tassadar/reports/tassadar_post_article_bounded_weighted_plugin_platform_closeout_audit_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_authority_promotion_publication_and_trust_tier_gate.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article.canonical_computational_model.statement.v1"
' fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_conformance_sandbox_and_benchmark_harness.eval_report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article.canonical_computational_model.statement.v1"
' fixtures/tassadar/reports/tassadar_post_article_plugin_conformance_sandbox_and_benchmark_harness_eval_report.json >/dev/null

jq -e '
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and (.reserved_capability_issue_ids | index("TAS-211")) != null
  and (.reserved_capability_issue_ids | index("TAS-210")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
