#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

manifest_path="fixtures/tassadar/sources/tassadar_article_frontend_compiler_envelope_v1.json"
report_path="fixtures/tassadar/reports/tassadar_article_frontend_compiler_envelope_report.json"
summary_path="fixtures/tassadar/reports/tassadar_article_frontend_compiler_envelope_summary.json"
acceptance_gate_path="fixtures/tassadar/reports/tassadar_article_equivalence_acceptance_gate_report.json"

if [[ "${REGENERATE:-0}" == "1" ]]; then
  cargo run -p psionic-compiler --example tassadar_article_frontend_compiler_envelope_manifest
  cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
  cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
  cargo run -p psionic-eval --example tassadar_article_frontend_compiler_envelope_report
  cargo run -p psionic-research --example tassadar_article_frontend_compiler_envelope_summary
fi

jq -e '
  .manifest_id == "tassadar.article_frontend_compiler_envelope.v1"
  and (.admitted_source_rows | length) == 8
  and (.disallowed_rows | length) == 6
' "$manifest_path" >/dev/null

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-176"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-185A")
  and .manifest_check.manifest_green == true
  and .compile_matrix_tie.green == true
  and .admitted_case_green_count == 8
  and .refusal_probe_green_count == 6
  and .toolchain_identity_green == true
  and .refusal_taxonomy_green == true
  and .envelope_manifest_green == true
  and .article_equivalence_green == false
' "$report_path" >/dev/null

jq -e '
  .tied_requirement_id == "TAS-176"
  and .tied_requirement_satisfied == true
  and .admitted_case_green_count == 8
  and .refusal_probe_green_count == 6
  and .toolchain_identity_green == true
  and .refusal_taxonomy_green == true
  and .envelope_manifest_green == true
  and .article_equivalence_green == false
' "$summary_path" >/dev/null

jq -e '
  .closed_required_issue_count == 35
  and .passed_required_requirement_count == 37
  and (.blocked_issue_ids[0] == "TAS-185A")
  and (.green_requirement_ids | index("TAS-176")) != null
  and (.green_requirement_ids | index("TAS-177")) != null
  and (.green_requirement_ids | index("TAS-178")) != null
  and (.green_requirement_ids | index("TAS-179")) != null
  and (.green_requirement_ids | index("TAS-179A")) != null
  and (.green_requirement_ids | index("TAS-180")) != null
  and (.green_requirement_ids | index("TAS-181")) != null
  and (.green_requirement_ids | index("TAS-182")) != null
  and (.green_requirement_ids | index("TAS-183")) != null
  and (.green_requirement_ids | index("TAS-184")) != null
  and (.green_requirement_ids | index("TAS-184A")) != null
' "$acceptance_gate_path" >/dev/null
