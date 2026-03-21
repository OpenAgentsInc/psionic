#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_demo_benchmark_equivalence_gate_report
cargo run -p psionic-research --example tassadar_article_demo_benchmark_equivalence_gate_summary
cargo run -p psionic-eval --example tassadar_article_runtime_closeout_report
cargo run -p psionic-eval --example tassadar_article_fast_route_throughput_floor_report
cargo run -p psionic-eval --example tassadar_execution_checkpoint_report
cargo run -p psionic-eval --example tassadar_spill_tape_store_report
cargo run -p psionic-eval --example tassadar_effect_safe_resume_report
cargo run -p psionic-eval --example tassadar_dynamic_memory_resume_report
cargo run -p psionic-eval --example tassadar_article_single_run_no_spill_closure_report
cargo run -p psionic-research --example tassadar_article_single_run_no_spill_closure_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-183"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-185")
  and .benchmark_prerequisite.tied_requirement_id == "TAS-182"
  and .benchmark_prerequisite.article_demo_benchmark_equivalence_gate_green == true
  and .operator_envelope.operator_envelope_green == true
  and .horizon_review.deterministic_exactness_green == true
  and .step_consistency_review.consistency_green == true
  and .context_sensitivity_review.context_sensitivity_green == true
  and .boundary_perturbation_review.perturbation_negative_control_green == true
  and .stochastic_mode_review.stochastic_mode_robustness_green == true
  and .binding_review.binding_green == true
  and .single_run_no_spill_closure_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_single_run_no_spill_closure_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-183"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-185"
  and .deterministic_exactness_green == true
  and .step_consistency_green == true
  and .context_sensitivity_green == true
  and .perturbation_negative_control_green == true
  and .stochastic_mode_robustness_green == true
  and .single_run_no_spill_closure_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_single_run_no_spill_closure_summary.json >/dev/null
