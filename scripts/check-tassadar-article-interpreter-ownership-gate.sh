#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_canonical_transformer_stack_boundary_report
cargo run -p psionic-eval --example tassadar_article_interpreter_breadth_suite_gate_report
cargo run -p psionic-eval --example tassadar_article_transformer_reference_linear_exactness_gate_report
cargo run -p psionic-eval --example tassadar_article_representation_invariance_gate_report
cargo run -p psionic-eval --example tassadar_article_transformer_generalization_gate_report
cargo run -p psionic-eval --example tassadar_article_evaluation_independence_audit_report
cargo run -p psionic-eval --example tassadar_article_transformer_weight_lineage_report
cargo run -p psionic-serve --example tassadar_direct_model_weight_execution_proof_report
cargo run -p psionic-eval --example tassadar_article_single_run_no_spill_closure_report
cargo run -p psionic-eval --example tassadar_article_interpreter_ownership_gate_report
cargo run -p psionic-research --example tassadar_article_interpreter_ownership_gate_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-184"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-185")
  and .canonical_boundary_report.boundary_contract_green == true
  and .generic_direct_proof_review.generic_direct_proof_suite_green == true
  and .generic_direct_proof_review.direct_case_count == 6
  and (.generic_direct_proof_review.synthetic_receipt_case_ids | length) == 3
  and .breadth_conformance_matrix.conformance_matrix_green == true
  and .breadth_conformance_matrix.green_family_count == 8
  and .route_purity_review.route_purity_green == true
  and .computation_mapping_report.stable_across_runs == true
  and .weight_perturbation_review.all_interventions_show_sensitivity == true
  and (.weight_perturbation_review.intervention_rows | length) == 4
  and .binding_review.ownership_gate_green == true
  and .interpreter_ownership_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_interpreter_ownership_gate_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-184"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-185"
  and .generic_direct_proof_suite_green == true
  and .generic_direct_proof_case_count == 6
  and .breadth_conformance_matrix_green == true
  and .green_family_count == 8
  and .route_purity_green == true
  and .mapping_stable_across_runs == true
  and .perturbation_sensitivity_green == true
  and .interpreter_ownership_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_interpreter_ownership_gate_summary.json >/dev/null
