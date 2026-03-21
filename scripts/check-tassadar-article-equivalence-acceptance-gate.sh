#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report

jq -e '
  .acceptance_status == "blocked"
  and .article_equivalence_green == false
  and .blocker_matrix_contract_green == true
  and .prerequisite_transformer_boundary_green == true
  and .blocker_matrix_article_equivalence_green == false
  and .public_claim_allowed == false
  and .closed_required_issue_count == 29
  and .passed_required_requirement_count == 31
  and .open_blocker_count == 4
  and ((.green_requirement_ids | index("TAS-158")) != null)
  and ((.green_requirement_ids | index("TAS-159")) != null)
  and ((.green_requirement_ids | index("TAS-160")) != null)
  and ((.green_requirement_ids | index("TAS-161")) != null)
  and ((.green_requirement_ids | index("TAS-162")) != null)
  and ((.green_requirement_ids | index("TAS-163")) != null)
  and ((.green_requirement_ids | index("TAS-176")) != null)
  and ((.green_requirement_ids | index("TAS-177")) != null)
  and ((.green_requirement_ids | index("TAS-178")) != null)
  and ((.green_requirement_ids | index("TAS-179")) != null)
  and ((.green_requirement_ids | index("TAS-179A")) != null)
  and ((.failed_requirement_ids | index("article_equivalence_blockers_closed")) != null)
  and ((.optional_open_issue_ids | index("TAS-R1")) != null)
  and ((.blocked_issue_ids | length) > 0)
  and (.blocked_issue_ids[0] == "TAS-181")
' fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json >/dev/null
