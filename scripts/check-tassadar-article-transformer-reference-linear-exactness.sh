#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_fixture_transformer_parity_report
cargo run -p psionic-serve --example tassadar_direct_model_weight_execution_proof_report
cargo run -p psionic-eval --example tassadar_article_transformer_reference_linear_exactness_gate_report
cargo run -p psionic-research --example tassadar_article_transformer_reference_linear_exactness_summary

jq -e '
  .reference_linear_exactness_green == true
  and .article_equivalence_green == false
  and .declared_case_count == 13
  and .exact_case_count == 13
  and .mismatch_case_count == 0
  and .refused_case_count == 0
  and .within_transformer_context_window_case_count == 4
  and .direct_model_weight_proof_case_count == 3
  and .parity_review.passed == true
  and .direct_proof_review.passed == true
  and ((.acceptance_gate_tie.tied_requirement_id) == "TAS-171A")
  and ((.acceptance_gate_tie.tied_requirement_satisfied) == true)
  and ((.mismatch_case_ids | length) == 0)
  and ((.refused_case_ids | length) == 0)
' fixtures/tassadar/reports/tassadar_article_transformer_reference_linear_exactness_gate_report.json >/dev/null
