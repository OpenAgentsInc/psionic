#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -q -p psionic-train --example tassadar_default_train_lane_fixtures
cargo test -q -p psionic-train tassadar_default_train_lane -- --nocapture

jq -e '
  .schema_version == "tassadar.default_train_lane_contract.v1"
  and .lane_id == "tassadar_article_transformer_trace_bound_trained_v0"
  and .launcher_path == "./TRAIN_TASSADAR"
  and .training_run_profile == "bounded_article_weight_production"
  and .hardware_profile_id == "cpu_reference"
  and .run_root_family == "fixtures/tassadar/runs/tassadar_article_transformer_weight_production_v1"
  and .evidence_family == "train.tassadar.article_transformer.weight_production"
  and .restart_posture == "restart_from_trace_bound_base_v0"
  and .promotion_target_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and (.checker_refs | any(. == "scripts/check-tassadar-default-train-lane.sh"))
  and (.checker_refs | any(. == "scripts/check-tassadar-acceptance.sh"))
' fixtures/tassadar/operator/tassadar_default_train_lane_contract_v1.json >/dev/null
