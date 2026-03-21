#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_universal_machine_proof_rebinding_report
cargo run -p psionic-research --example tassadar_post_article_universal_machine_proof_rebinding_summary
cargo test -p psionic-provider proof_rebinding_receipt_projects_summary -- --nocapture

jq -e '
  .proof_rebinding_status == "green"
  and .proof_transport_audit_complete == true
  and .proof_rebinding_complete == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .carrier_split_publication_complete == true
  and .universality_witness_suite_reissued == false
  and .universal_substrate_gate_allowed == false
  and ((.supporting_material_rows | length) == 9)
  and ((.proof_transport_receipt_rows | length) == 2)
  and ((.rebound_encoding_ids | length) == 2)
  and ((.validation_rows | length) == 9)
  and (.deferred_issue_ids == ["TAS-191"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_report.json >/dev/null

jq -e '
  .proof_rebinding_status == "green"
  and .proof_transport_audit_complete == true
  and .proof_rebinding_complete == true
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and ((.rebound_encoding_ids | length) == 2)
  and (.deferred_issue_ids == ["TAS-191"])
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_universal_machine_proof_rebinding_summary.json >/dev/null
