#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_interpreter_breadth_envelope_report
cargo run -p psionic-research --example tassadar_article_interpreter_breadth_envelope_summary
cargo run -p psionic-eval --example tassadar_article_interpreter_breadth_suite_gate_report
cargo run -p psionic-research --example tassadar_article_interpreter_breadth_suite_gate_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-179A"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-180")
  and .green_family_count == 8
  and .contract_check.contract_green == true
  and ((.family_checks | map(select(.green == true)) | length) == 8)
  and .breadth_gate_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_interpreter_breadth_suite_gate_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-179A"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-180"
  and .green_family_count == 8
  and .required_family_count == 8
  and .breadth_gate_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_interpreter_breadth_suite_gate_summary.json >/dev/null
