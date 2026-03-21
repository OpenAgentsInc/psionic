#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_hungarian_demo_parity_report
cargo run -p psionic-research --example tassadar_article_hungarian_demo_parity_summary
cargo run -p psionic-eval --example tassadar_article_hard_sudoku_benchmark_closure_report
cargo run -p psionic-research --example tassadar_article_hard_sudoku_benchmark_closure_summary
cargo run -p psionic-eval --example tassadar_article_demo_benchmark_equivalence_gate_report
cargo run -p psionic-research --example tassadar_article_demo_benchmark_equivalence_gate_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-182"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-186")
  and .hungarian_review.hungarian_demo_parity_green == true
  and .benchmark_review.named_arto_parity_green == true
  and .benchmark_review.benchmark_wide_sudoku_parity_green == true
  and .binding_review.binding_green == true
  and .article_demo_benchmark_equivalence_gate_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_demo_benchmark_equivalence_gate_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-182"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-186"
  and .hungarian_demo_parity_green == true
  and .named_arto_parity_green == true
  and .benchmark_wide_sudoku_parity_green == true
  and .binding_green == true
  and .article_demo_benchmark_equivalence_gate_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_demo_benchmark_equivalence_gate_summary.json >/dev/null
