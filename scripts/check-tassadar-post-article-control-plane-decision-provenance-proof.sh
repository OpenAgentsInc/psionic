#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_control_plane_decision_provenance_proof_report
cargo run -p psionic-research --example tassadar_post_article_control_plane_decision_provenance_proof_summary
cargo test -p psionic-provider control_plane_receipt_projects_summary -- --nocapture

jq -e '
  .control_plane_ownership_status == "green"
  and .control_plane_ownership_green == true
  and .replay_posture_green == true
  and .decision_provenance_proof_complete == true
  and .carrier_split_publication_complete == false
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and ((.supporting_material_rows | length) == 7)
  and ((.decision_binding_rows | length) == 3)
  and ((.hidden_control_channel_rows | length) == 6)
  and ((.validation_rows | length) == 7)
  and (.deferred_issue_ids == ["TAS-189"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_report.json >/dev/null

jq -e '
  .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .control_plane_ownership_green == true
  and .replay_posture_green == true
  and .decision_provenance_proof_complete == true
  and .carrier_split_publication_complete == false
  and (.deferred_issue_ids == ["TAS-189"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_control_plane_decision_provenance_proof_summary.json >/dev/null
