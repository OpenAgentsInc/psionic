#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_bundle
cargo run -p psionic-sandbox --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report
cargo run -p psionic-research --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary
cargo run -p psionic-serve --example tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication
cargo run -p psionic-runtime --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_bundle
cargo run -p psionic-eval --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_report
cargo run -p psionic-research --example tassadar_post_article_plugin_invocation_receipts_and_replay_classes_summary
cargo test -p psionic-provider post_article_plugin_runtime_api_receipt_projects_summary -- --nocapture
cargo test -p psionic-provider post_article_plugin_invocation_receipts_receipt_projects_summary -- --nocapture
cargo test -p psionic-serve post_article_plugin_runtime_api_publication_keeps_served_surface_blocked -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article_plugin_invocation_receipts_and_replay_classes.runtime_bundle.v1"
  and .profile_id == "tassadar.plugin_runtime.invocation_receipts.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and .packet_abi_version == "packet.v1"
  and ((.receipt_field_rows | length) == 18)
  and ((.replay_class_rows | length) == 4)
  and ((.failure_class_rows | length) == 12)
  and ((.case_receipts | length) == 7)
  and .exact_success_case_count == 1
  and .exact_refusal_case_count == 4
  and .exact_failure_case_count == 2
  and .challenge_bound_case_count == 3
' fixtures/tassadar/runs/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_v1/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_bundle.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_invocation_receipts_and_replay_classes.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .machine_identity_binding.engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and .machine_identity_binding.invocation_receipt_profile_id == "tassadar.plugin_runtime.invocation_receipts.v1"
  and ((.dependency_rows | length) == 5)
  and ((.receipt_identity_rows | length) == 18)
  and ((.replay_class_rows | length) == 4)
  and ((.failure_class_rows | length) == 12)
  and ((.validation_rows | length) == 9)
  and .operator_internal_only_posture == true
  and .receipt_identity_frozen == true
  and .resource_summary_required == true
  and .failure_lattice_frozen == true
  and .deterministic_replay_class_frozen == true
  and .snapshot_replay_class_frozen == true
  and .operator_replay_only_class_frozen == true
  and .publication_refusal_class_frozen == true
  and .route_evidence_binding_required == true
  and .challenge_receipt_binding_required == true
  and .replay_retry_propagation_typed == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == ["TAS-202"])
' fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_invocation_receipts_and_replay_classes.report.v1"
  and .packet_abi_version == "packet.v1"
  and .host_owned_runtime_api_id == "tassadar.plugin_runtime.host_owned_api.v1"
  and .engine_abstraction_id == "tassadar.plugin_runtime.engine_abstraction.v1"
  and .invocation_receipt_profile_id == "tassadar.plugin_runtime.invocation_receipts.v1"
  and .contract_status == "green"
  and .dependency_row_count == 5
  and .receipt_identity_row_count == 18
  and .replay_class_row_count == 4
  and .failure_class_row_count == 12
  and .validation_row_count == 9
  and (.deferred_issue_ids == ["TAS-202"])
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_invocation_receipts_and_replay_classes_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
' fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_summary.json >/dev/null

jq -e '
  .publication_id == "tassadar.post_article_plugin_runtime_api_and_engine_abstraction.publication.v1"
  and (.served_plugin_surface_ids == [])
  and (.blocked_by == [])
  and .plugin_publication_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_runtime_api_and_engine_abstraction_publication.json >/dev/null
