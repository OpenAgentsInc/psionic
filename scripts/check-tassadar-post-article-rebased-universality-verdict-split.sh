#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_rebased_universality_verdict_split_report
cargo run -p psionic-research --example tassadar_post_article_rebased_universality_verdict_split_summary
cargo run -p psionic-serve --example tassadar_post_article_rebased_universality_verdict_publication
cargo test -p psionic-provider post_article_rebased_universality_verdict_receipt_projects_publication -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_rebased_universality_verdict_split.report.v1"
  and .verdict_split_status == "theory_green_operator_green_served_suppressed"
  and .verdict_split_green == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .current_served_internal_compute_profile_id == "tassadar.internal_compute.article_closeout.v1"
  and ((.supporting_material_rows | length) == 7)
  and ((.supporting_material_rows | map(select(.material_class == "proof_carrying" and .satisfied == true)) | length) == 6)
  and ((.supporting_material_rows | map(select(.material_class == "observational_context")) | length) == 1)
  and ((.validation_rows | length) == 8)
  and .theory_green == true
  and .operator_green == true
  and .served_green == false
  and (.deferred_issue_ids == ["TAS-195"])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_split_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_rebased_universality_verdict_split.report.v1"
  and .verdict_split_status == "theory_green_operator_green_served_suppressed"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .current_served_internal_compute_profile_id == "tassadar.internal_compute.article_closeout.v1"
  and .supporting_material_row_count == 7
  and .validation_row_count == 8
  and .theory_green == true
  and .operator_green == true
  and .served_green == false
  and (.deferred_issue_ids == ["TAS-195"])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_split_summary.json >/dev/null

jq -e '
  .publication_id == "tassadar.post_article_rebased_universality_verdict.publication.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .current_served_internal_compute_profile_id == "tassadar.internal_compute.article_closeout.v1"
  and .theory_green == true
  and .operator_green == true
  and .served_green == false
  and ((.operator_allowed_profile_ids | length) == 3)
  and ((.served_blocked_by | length) == 3)
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_rebased_universality_verdict_publication.json >/dev/null
