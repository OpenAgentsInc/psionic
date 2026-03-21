#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-serve --example tassadar_article_hungarian_demo_fast_route_session_artifact
cargo run -p psionic-serve --example tassadar_article_hungarian_demo_fast_route_hybrid_workflow_artifact
cargo run -p psionic-eval --example tassadar_article_hungarian_demo_parity_report
cargo run -p psionic-research --example tassadar_article_hungarian_demo_parity_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-180"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-186")
  and .frontend_review.row_green == true
  and .no_tool_proof_review.no_tool_proof_green == true
  and .fast_route_session_review.fast_route_direct_green == true
  and .fast_route_hybrid_review.fast_route_direct_green == true
  and .throughput_review.declared_floor_green == true
  and .binding_review.binding_green == true
  and .hungarian_demo_parity_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_hungarian_demo_parity_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-180"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-186"
  and .session_fast_route_green == true
  and .hybrid_fast_route_green == true
  and .throughput_floor_green == true
  and .no_tool_proof_green == true
  and .binding_green == true
  and .hungarian_demo_parity_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_hungarian_demo_parity_summary.json >/dev/null
