#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-data --example tassadar_multi_plugin_trace_corpus_bundle
cargo test -p psionic-data multi_plugin_trace_corpus_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.multi_plugin_trace_corpus.bundle.v1"
  and ((.source_bundle_rows | length) == 3)
  and ((.projected_tool_schema_rows | length) == 4)
  and ((.trace_records | length) == 6)
  and ((.parity_matrix.workflow_parity_rows | length) == 2)
  and (.parity_matrix.explicit_disagreement_count > 0)
  and (.bootstrap_contract.contract_id == "psionic.tassadar.multi_plugin_training_bootstrap.v1")
  and (.bootstrap_contract.bootstrap_ready == true)
  and (.trace_records | any(.lane_id == "deterministic_workflow"))
  and (.trace_records | any(.lane_id == "router_responses"))
  and (.trace_records | any(.lane_id == "apple_fm_session"))
  and (.parity_matrix.workflow_parity_rows | any(
    .workflow_case_id == "starter_plugin.web_content_success.v1"
    and (.disagreement_rows | length) > 0
    and (.step_parity_rows | any(.agreement_class == "digest_drift"))
  ))
' fixtures/tassadar/datasets/tassadar_multi_plugin_trace_corpus_v1/tassadar_multi_plugin_trace_corpus_bundle.json >/dev/null
