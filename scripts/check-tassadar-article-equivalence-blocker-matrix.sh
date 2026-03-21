#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report

jq -e '
  .matrix_contract_green == true
  and .article_equivalence_green == false
  and .prerequisite_transformer_boundary_green == true
  and .all_required_categories_present == true
  and .all_blockers_have_article_line_provenance == true
  and .all_blockers_covered_by_issue_map == true
  and .all_later_issues_covered == true
  and .all_issue_refs_point_to_known_blockers == true
  and .blocker_count == 7
  and .open_blocker_count == 3
  and ((.issue_coverage_rows | map(select(.issue_id == "TAS-176" and .issue_state == "closed")) | length) == 1)
  and ((.issue_coverage_rows | map(select(.issue_id == "TAS-177" and .issue_state == "closed")) | length) == 1)
  and ((.issue_coverage_rows | map(select(.issue_id == "TAS-178" and .issue_state == "closed")) | length) == 1)
  and ((.issue_coverage_rows | map(select(.issue_id == "TAS-179" and .issue_state == "closed")) | length) == 1)
  and ((.issue_coverage_rows | map(select(.issue_id == "TAS-179A" and .issue_state == "closed")) | length) == 1)
  and ((.issue_coverage_rows | map(select(.issue_id == "TAS-182" and .issue_state == "closed")) | length) == 1)
' fixtures/tassadar/reports/tassadar_article_equivalence_blocker_matrix_report.json >/dev/null
