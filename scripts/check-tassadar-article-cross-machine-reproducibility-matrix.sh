#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_fast_route_architecture_selection_report
cargo run -p psionic-eval --example tassadar_article_fast_route_exactness_report
cargo run -p psionic-eval --example tassadar_article_demo_benchmark_equivalence_gate_report
cargo run -p psionic-eval --example tassadar_article_single_run_no_spill_closure_report
cargo run -p psionic-eval --example tassadar_article_fast_route_throughput_floor_report
cargo run -p psionic-eval --example tassadar_article_cross_machine_reproducibility_matrix_report
cargo run -p psionic-research --example tassadar_article_cross_machine_reproducibility_matrix_summary
cargo run -p psionic-serve --example tassadar_article_cross_machine_reproducibility_publication

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-185"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-185A")
  and .machine_matrix_review.current_host_measured_green == true
  and (.machine_matrix_review.supported_machine_class_ids | length) == 2
  and .machine_matrix_review.machine_class_alignment_green == true
  and .route_stability_review.route_stability_green == true
  and .demo_review.demo_benchmark_equivalence_green == true
  and .long_horizon_review.deterministic_exactness_green == true
  and .long_horizon_review.single_run_no_spill_closure_green == true
  and .throughput_drift_review.drift_policy_green == true
  and .throughput_floor_stability_green == true
  and .stochastic_mode_review.stochastic_mode_supported == false
  and .stochastic_mode_review.out_of_scope == true
  and .deterministic_mode_green == true
  and .reproducibility_matrix_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_matrix_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-185"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-185A"
  and (.supported_machine_class_ids | length) == 2
  and .deterministic_mode_green == true
  and .throughput_floor_stability_green == true
  and .stochastic_mode_supported == false
  and .stochastic_mode_out_of_scope == true
  and .reproducibility_matrix_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_matrix_summary.json >/dev/null

jq -e '
  (.supported_machine_class_ids | length) == 2
  and .selected_decode_mode == "hull_cache"
  and .deterministic_mode_green == true
  and .throughput_floor_stability_green == true
  and .stochastic_mode_out_of_scope == true
  and .reproducibility_matrix_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_publication.json >/dev/null
