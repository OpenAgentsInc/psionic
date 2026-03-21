#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_interpreter_breadth_envelope_report
cargo run -p psionic-research --example tassadar_article_interpreter_breadth_envelope_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-179"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-185")
  and .current_floor_green_count == 2
  and .declared_required_family_green_count == 3
  and .research_only_family_green_count == 1
  and .explicit_out_of_envelope_green_count == 7
  and .contract_check.contract_green == true
  and .envelope_contract_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_interpreter_breadth_envelope_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-179"
  and .tied_requirement_satisfied == true
  and .current_floor_green_count == 2
  and .declared_required_family_green_count == 3
  and .research_only_family_green_count == 1
  and .explicit_out_of_envelope_green_count == 7
  and .envelope_contract_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_interpreter_breadth_envelope_summary.json >/dev/null
