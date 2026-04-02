#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

rustflags="${RUSTFLAGS:-}"
if [[ -n "${rustflags}" ]]; then
  rustflags="${rustflags} -Awarnings"
else
  rustflags="-Awarnings"
fi

env RUSTFLAGS="${rustflags}" cargo run -q -p psionic-train --example tassadar_default_train_rehearsal_fixtures
env RUSTFLAGS="${rustflags}" cargo test -q -p psionic-train tassadar_default_train_rehearsal -- --nocapture

jq -e '
  .schema_version == "tassadar.default_train_rehearsal_bundle.v1"
  and .lane_id == "tassadar_article_transformer_trace_bound_trained_v0"
  and .launcher_path == "./TRAIN_TASSADAR"
  and .launcher_surface_id == "tassadar_train_start"
  and .launch_manifest_ref == "manifests/launch_manifest.json"
  and .current_run_status_ref == "status/current_run_status.json"
  and .retained_summary_ref == "status/retained_summary.json"
  and .promotion_evidence_ref == "promotion/promotion_target_evidence.json"
  and (.checker_receipt_refs | index("checker/default_train_lane_contract_check.json"))
  and (.checker_receipt_refs | index("checker/acceptance_check.json"))
  and (.closeout_gates | length) == 4
  and (.can_now_claim | length) >= 2
  and (.still_out_of_scope | length) >= 3
' fixtures/tassadar/operator/tassadar_default_train_rehearsal_bundle_v1.json >/dev/null
