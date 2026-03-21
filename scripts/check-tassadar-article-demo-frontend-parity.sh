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
cargo run -p psionic-eval --example tassadar_article_demo_frontend_parity_report
cargo run -p psionic-research --example tassadar_article_demo_frontend_parity_summary

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-178"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and .compiled_demo_count == 2
  and .green_demo_count == 2
  and .refusal_probe_green_count == 2
  and .source_compile_receipt_parity_green == true
  and .workload_identity_parity_green == true
  and .unsupported_variant_refusal_green == true
  and .demo_frontend_parity_green == true
  and .article_equivalence_green == false
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-185")
' fixtures/tassadar/reports/tassadar_article_demo_frontend_parity_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-178"
  and .tied_requirement_satisfied == true
  and .compiled_demo_count == 2
  and .green_demo_count == 2
  and .refusal_probe_green_count == 2
  and .source_compile_receipt_parity_green == true
  and .workload_identity_parity_green == true
  and .unsupported_variant_refusal_green == true
  and .demo_frontend_parity_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_demo_frontend_parity_summary.json >/dev/null
