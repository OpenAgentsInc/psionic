#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-serve --example tassadar_article_transformer_minimal_frontier_report

jq -e '
  .issue_id == "TAS-R1"
  and .schema_version == 1
  and (.candidate_reports | length) == 6
  and .acceptance_gate_article_equivalence_green == true
  and .canonical_optional_issue_still_open == true
  and .frontier_green == false
  and .minimal_successful_candidate_id == null
  and .minimal_successful_model_id == null
  and (.successful_candidate_ids | length) == 0
  and ([.candidate_reports[].stage_a_review.passed] | all)
  and ([.candidate_reports[].stage_b_review.passed] | all)
  and ([.candidate_reports[].stage_c_review.passed] | any | not)
  and ([.candidate_reports[].stage_c_review.direct_proof_review.passed] | all)
  and ([.candidate_reports[].stage_c_review.fast_route_selection_review.passed] | any | not)
  and ([.candidate_reports[].stage_c_review.throughput_floor_review.passed] | all)
' fixtures/tassadar/reports/tassadar_article_transformer_minimal_frontier_report.json >/dev/null
