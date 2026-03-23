#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-runtime --example tassadar_post_article_starter_plugin_workflow_controller_bundle
cargo test -p psionic-runtime starter_plugin_workflow_ -- --nocapture

jq -e '
  .bundle_id == "tassadar.post_article.starter_plugin_workflow_controller.bundle.v1"
  and .workflow_graph_id == "starter_flow.web_content_intake.v1"
  and ((.case_rows | length) == 3)
  and (.case_rows | any(.case_id == "web_content_intake_success" and .green == true and (.step_rows | length) == 6 and (.step_rows | any(.tool_name == "plugin_text_stats")) and (.final_artifact.text_stats_rows | length) == 1 and .stop_condition_id == "controller_stop.all_urls_processed"))
  and (.case_rows | any(.case_id == "web_content_intake_fetch_refusal" and .green == false and (.refusal_rows | length) == 1 and .stop_condition_id == "controller_stop.typed_refusal"))
  and (.case_rows | any(.case_id == "guest_artifact_echo_success" and .green == true and (.step_rows | length) == 1 and (.step_rows | any(.tool_name == "plugin_example_echo_guest")) and .stop_condition_id == "controller_stop.guest_artifact_echo_complete"))
' fixtures/tassadar/runs/tassadar_post_article_starter_plugin_workflow_controller_v1/tassadar_post_article_starter_plugin_workflow_controller_bundle.json >/dev/null
