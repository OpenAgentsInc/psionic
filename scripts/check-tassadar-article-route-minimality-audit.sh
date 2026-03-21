#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
cargo run -p psionic-eval --example tassadar_article_fast_route_architecture_selection_report
cargo run -p psionic-eval --example tassadar_article_single_run_no_spill_closure_report
cargo run -p psionic-eval --example tassadar_article_interpreter_ownership_gate_report
cargo run -p psionic-eval --example tassadar_article_kv_activation_discipline_audit_report
cargo run -p psionic-eval --example tassadar_article_cross_machine_reproducibility_matrix_report
cargo run -p psionic-eval --example tassadar_article_route_minimality_audit_report
cargo run -p psionic-research --example tassadar_article_route_minimality_audit_summary
cargo run -p psionic-serve --example tassadar_article_route_minimality_publication_verdict

jq -e '
  .acceptance_gate_tie.tied_requirement_id == "TAS-185A"
  and .acceptance_gate_tie.tied_requirement_satisfied == true
  and (.acceptance_gate_tie.blocked_issue_ids[0] == "TAS-186")
  and .canonical_claim_route_review.canonical_claim_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .canonical_claim_route_review.selected_decode_mode == "hull_cache"
  and .continuation_boundary_review.checkpoint_restore_allowed == false
  and .continuation_boundary_review.spill_tape_extension_allowed == false
  and .continuation_boundary_review.external_persisted_continuation_allowed == false
  and .continuation_boundary_review.hidden_reentry_allowed == false
  and .continuation_boundary_review.implicit_segmentation_allowed == false
  and .continuation_boundary_review.runtime_loop_unrolling_allowed == false
  and .continuation_boundary_review.perturbation_negative_control_green == true
  and .continuation_boundary_review.continuation_boundary_green == true
  and .execution_ownership_review.route_purity_green == true
  and .state_carrier_review.state_carrier_minimality_green == true
  and .orchestration_review.public_claim_route_excludes_planner_indirection == true
  and .orchestration_review.public_claim_route_excludes_hybrid_surface == true
  and .orchestration_review.extra_orchestration_layers_excluded == true
  and .operator_verdict_review.operator_verdict_green == true
  and .public_verdict_review.posture == "suppressed_pending_final_audit"
  and .public_verdict_review.public_verdict_green == false
  and .route_minimality_audit_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_route_minimality_audit_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-185A"
  and .tied_requirement_satisfied == true
  and .blocked_issue_frontier == "TAS-186"
  and .canonical_claim_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .selected_decode_mode == "tassadar.decode.hull_cache.v1"
  and .operator_verdict_green == true
  and .public_posture == "suppressed_pending_final_audit"
  and .public_verdict_green == false
  and .route_minimality_audit_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_route_minimality_audit_summary.json >/dev/null

jq -e '
  .canonical_claim_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .selected_decode_mode == "tassadar.decode.hull_cache.v1"
  and .operator_verdict_green == true
  and .public_posture == "suppressed_pending_final_audit"
  and (.public_blocked_issue_ids[0] == "TAS-186")
  and .route_minimality_audit_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_route_minimality_publication_verdict.json >/dev/null
