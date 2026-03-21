#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_frontend_compiler_envelope_report
cargo run -p psionic-research --example tassadar_article_frontend_compiler_envelope_summary
cargo run -p psionic-eval --example tassadar_article_frontend_corpus_compile_matrix_report
cargo run -p psionic-research --example tassadar_article_frontend_corpus_compile_matrix_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-177"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and .compiled_case_count == 11
  and .typed_refusal_case_count == 4
  and .toolchain_failure_case_count == 1
  and .lineage_green_count == 11
  and .refusal_green_count == 4
  and .toolchain_failure_green_count == 1
  and .category_coverage_green == true
  and .envelope_alignment_green == true
  and .compile_matrix_green == true
  and .article_equivalence_green == false
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-182")
' fixtures/tassadar/reports/tassadar_article_frontend_corpus_compile_matrix_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-177"
  and .tied_requirement_satisfied == true
  and .compiled_case_count == 11
  and .typed_refusal_case_count == 4
  and .toolchain_failure_case_count == 1
  and .category_coverage_green == true
  and .envelope_alignment_green == true
  and .compile_matrix_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_frontend_corpus_compile_matrix_summary.json >/dev/null
