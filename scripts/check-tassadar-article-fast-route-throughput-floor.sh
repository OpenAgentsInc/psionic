#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${REGENERATE:-0}" == "1" ]]; then
  cargo run -p psionic-eval --example tassadar_article_equivalence_blocker_matrix_report
  cargo run -p psionic-research --example tassadar_article_equivalence_blocker_matrix_summary
  cargo run -p psionic-eval --example tassadar_article_equivalence_acceptance_gate_report
  cargo run -p psionic-eval --example tassadar_article_fast_route_architecture_selection_report
  cargo run -p psionic-research --example tassadar_article_fast_route_architecture_selection_summary
  cargo run -p psionic-eval --example tassadar_article_fast_route_implementation_report
  cargo run -p psionic-research --example tassadar_article_fast_route_implementation_summary
  cargo run -p psionic-eval --example tassadar_article_fast_route_exactness_report
  cargo run -p psionic-research --example tassadar_article_fast_route_exactness_summary
  cargo run -p psionic-eval --example tassadar_article_runtime_closeout_report
  cargo run -p psionic-research --example tassadar_article_runtime_closeout_summary
  cargo run -p psionic-eval --example tassadar_article_cpu_reproducibility_report
  cargo run -p psionic-research --example tassadar_article_cpu_reproducibility_summary
  cargo run -p psionic-runtime --example tassadar_article_fast_route_throughput_bundle
  cargo run -p psionic-eval --example tassadar_article_fast_route_throughput_floor_report
  cargo run -p psionic-research --example tassadar_article_fast_route_throughput_floor_summary
fi

jq -e '
  .throughput_floor_green == true
  and .article_equivalence_green == false
  and ((.acceptance_gate_tie.tied_requirement_id) == "TAS-175")
  and ((.acceptance_gate_tie.tied_requirement_satisfied) == true)
  and ((.selection_prerequisite.selected_candidate_kind) == "hull_cache_runtime")
  and ((.selection_prerequisite.fast_route_selection_green) == true)
  and ((.exactness_prerequisite.exactness_green) == true)
  and ((.cross_machine_drift_review.allowed_floor_drift_bps) == 0)
  and ((.cross_machine_drift_review.drift_policy_green) == true)
  and ((.throughput_bundle.selected_candidate_kind) == "hull_cache_runtime")
  and ((.throughput_bundle.selected_decode_mode) == "hull_cache")
  and ((.throughput_bundle.throughput_floor_green) == true)
  and ((.throughput_bundle.demo_receipts | length) == 2)
  and ((.throughput_bundle.demo_receipts | map(select(.exactness_bps == 10000)) | length) == 2)
  and ((.throughput_bundle.demo_receipts | map(select(.selection_state == "direct")) | length) == 2)
  and ((.throughput_bundle.demo_receipts | map(select(.effective_decode_mode == "hull_cache")) | length) == 2)
  and ((.throughput_bundle.demo_receipts | map(select(.public_token_anchor_status == "passed")) | length) == 2)
  and ((.throughput_bundle.demo_receipts | map(select(.public_line_anchor_status == "passed")) | length) == 2)
  and ((.throughput_bundle.demo_receipts | map(select(.internal_token_floor_status == "passed")) | length) == 2)
  and ((.throughput_bundle.kernel_receipts | length) == 4)
  and ((.throughput_bundle.kernel_receipts | map(select(.exactness_bps == 10000)) | length) == 4)
  and ((.throughput_bundle.kernel_receipts | map(select(.floor_status == "passed")) | length) == 4)
' fixtures/tassadar/reports/tassadar_article_fast_route_throughput_floor_report.json >/dev/null

jq -e '
  .tied_requirement_id == "TAS-175"
  and .tied_requirement_satisfied == true
  and .selection_prerequisite_green == true
  and .exactness_prerequisite_green == true
  and .drift_policy_green == true
  and .demo_public_floor_pass_count == 2
  and .demo_internal_floor_pass_count == 2
  and .kernel_floor_pass_count == 4
  and .throughput_floor_green == true
  and .article_equivalence_green == false
' fixtures/tassadar/reports/tassadar_article_fast_route_throughput_floor_summary.json >/dev/null
