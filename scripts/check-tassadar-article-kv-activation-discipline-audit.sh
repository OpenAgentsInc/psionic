#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_kv_activation_discipline_audit_report
cargo run -p psionic-research --example tassadar_article_kv_activation_discipline_audit_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-184A"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-185")
  and .ownership_gate_green == true
  and .growth_report.cache_growth_scales_with_problem_size == true
  and .growth_report.dynamic_state_exceeds_weight_artifact_bytes == true
  and (.growth_report.feasible_constraint_case_ids | length) == 4
  and .sensitivity_review.cache_truncation_breaks_correctness == true
  and .sensitivity_review.cache_reset_breaks_correctness == true
  and .sensitivity_review.equivalent_behavior_survives_under_constrained_cache == false
  and .dominance_verdict.verdict == "mixed"
  and .kv_activation_discipline_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_kv_activation_discipline_audit_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-184A"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-185"
  and .ownership_gate_green == true
  and .feasible_constraint_case_count == 4
  and .dominance_verdict == "mixed"
  and .cache_growth_scales_with_problem_size == true
  and .dynamic_state_exceeds_weight_artifact_bytes == true
  and .cache_truncation_breaks_correctness == true
  and .cache_reset_breaks_correctness == true
  and .equivalent_behavior_survives_under_constrained_cache == false
  and .kv_activation_discipline_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_kv_activation_discipline_audit_summary.json >/dev/null
