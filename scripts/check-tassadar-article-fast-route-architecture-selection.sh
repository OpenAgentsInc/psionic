#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_fixture_transformer_parity_report
cargo run -p psionic-research --example tassadar_article_fixture_transformer_parity_summary
cargo run -p psionic-serve --example tassadar_direct_model_weight_execution_proof_report
cargo run -p psionic-eval --example tassadar_article_transformer_reference_linear_exactness_gate_report
cargo run -p psionic-research --example tassadar_article_transformer_reference_linear_exactness_summary
cargo run -p psionic-eval --example tassadar_article_transformer_generalization_gate_report
cargo run -p psionic-research --example tassadar_article_transformer_generalization_summary
cargo run -p psionic-eval --example tassadar_article_evaluation_independence_audit_report
cargo run -p psionic-research --example tassadar_article_evaluation_independence_summary
cargo run -p psionic-eval --example tassadar_article_fast_route_architecture_selection_report
cargo run -p psionic-research --example tassadar_article_fast_route_architecture_selection_summary

jq -e '
  .fast_route_selection_green == true
  and .article_equivalence_green == false
  and ((.acceptance_gate_tie.tied_requirement_id) == "TAS-172")
  and ((.acceptance_gate_tie.tied_requirement_satisfied) == true)
  and ((.selected_candidate_kind) == "hull_cache_runtime")
  and (.transformer_model_route_anchor_review.passed == true)
  and ((.routeability_checks | map(select(.candidate_kind == "hull_cache_runtime")) | length) == 1)
  and ((.routeability_checks[] | select(.candidate_kind == "hull_cache_runtime") | .routeable) == true)
  and ((.routeability_checks[] | select(.candidate_kind == "hull_cache_runtime") | .direct_module_class_count) == 4)
  and ((.routeability_checks[] | select(.candidate_kind == "hull_cache_runtime") | .fallback_module_class_count) == 2)
  and ((.candidate_verdicts[] | select(.candidate_kind == "hull_cache_runtime") | .selected) == true)
' fixtures/tassadar/reports/tassadar_article_fast_route_architecture_selection_report.json >/dev/null
