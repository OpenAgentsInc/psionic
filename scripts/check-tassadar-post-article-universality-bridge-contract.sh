#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo test -p psionic-provider post_article_universality_bridge_contract -- --nocapture

jq -e '
  .bridge_status == "green"
  and .bridge_contract_green == true
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and .carrier_topology == "explicit_split_across_direct_and_resumable_lanes"
  and .bridge_machine_identity.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .bridge_machine_identity.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .bridge_machine_identity.continuation_contract_id == "tassadar.tcm_v1.runtime_contract.report.v1"
  and ((.historical_binding_rows | length) == 4)
  and ((.validation_rows | length) == 8)
  and (.reserved_later_invariant_ids == ["choice_set_integrity", "resource_transparency", "scheduling_ownership"])
  and ((.reservation_hook_rows | length) == 4)
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_report.json >/dev/null

jq -e '
  .bridge_machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .carrier_topology == "explicit_split_across_direct_and_resumable_lanes"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .bridge_contract_green == true
  and .rebase_claim_allowed == false
  and .plugin_capability_claim_allowed == false
  and .served_public_universality_allowed == false
  and (.reserved_capability_issue_ids | index("TAS-195")) != null
  and (.reserved_capability_issue_ids | index("TAS-211")) != null
  and (.reserved_capability_issue_ids | index("TAS-206")) == null
  and (.reserved_capability_issue_ids | index("TAS-207")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null
