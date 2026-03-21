#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-runtime --example tassadar_article_hard_sudoku_benchmark_bundle
cargo run -p psionic-serve --example tassadar_article_hard_sudoku_fast_route_session_artifact
cargo run -p psionic-serve --example tassadar_article_hard_sudoku_fast_route_hybrid_workflow_artifact
cargo run -p psionic-eval --example tassadar_article_hard_sudoku_benchmark_closure_report
cargo run -p psionic-research --example tassadar_article_hard_sudoku_benchmark_closure_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-181"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-183")
  and .manifest_review.manifest_green == true
  and .frontend_review.row_green == true
  and .no_tool_proof_review.no_tool_proof_green == true
  and .fast_route_session_review.suite_green == true
  and .fast_route_hybrid_review.suite_green == true
  and .runtime_review.suite_green == true
  and .named_arto_green == true
  and .hard_sudoku_suite_green == true
  and .binding_review.binding_green == true
  and .hard_sudoku_benchmark_closure_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_hard_sudoku_benchmark_closure_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-181"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-183"
  and .named_arto_green == true
  and .hard_sudoku_suite_green == true
  and .session_fast_route_green == true
  and .hybrid_fast_route_green == true
  and .runtime_suite_green == true
  and .binding_green == true
  and .hard_sudoku_benchmark_closure_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_hard_sudoku_benchmark_closure_summary.json >/dev/null
