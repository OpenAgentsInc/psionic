#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_dir="${repo_root}/fixtures/psion/a1_minimal_distributed_lm"
report_path="${fixture_dir}/trusted_aggregation_report_v1.json"
checkpoint_path="${fixture_dir}/aggregate_checkpoint_v1.json"
promotion_path="${fixture_dir}/promotion_receipt_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/a1_minimal_distributed_lm_trusted_aggregation.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

cargo run -q -p psionic-train --example a1_minimal_distributed_lm_trusted_aggregation_fixture -- "${tmpdir}" >/dev/null

generated_dir="${tmpdir}/fixtures/psion/a1_minimal_distributed_lm"
generated_report_path="${generated_dir}/trusted_aggregation_report_v1.json"
generated_checkpoint_path="${generated_dir}/aggregate_checkpoint_v1.json"
generated_promotion_path="${generated_dir}/promotion_receipt_v1.json"

python3 - "${report_path}" "${generated_report_path}" "${checkpoint_path}" "${generated_checkpoint_path}" "${promotion_path}" "${generated_promotion_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
generated_report_path = Path(sys.argv[2])
checkpoint_path = Path(sys.argv[3])
generated_checkpoint_path = Path(sys.argv[4])
promotion_path = Path(sys.argv[5])
generated_promotion_path = Path(sys.argv[6])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

report = json.loads(report_path.read_text(encoding="utf-8"))
generated_report = json.loads(generated_report_path.read_text(encoding="utf-8"))
checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
generated_checkpoint = json.loads(generated_checkpoint_path.read_text(encoding="utf-8"))
promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
generated_promotion = json.loads(generated_promotion_path.read_text(encoding="utf-8"))

if report != generated_report:
    fail("A1 minimal distributed LM trusted aggregation report drifted from generator output")
if checkpoint != generated_checkpoint:
    fail("A1 minimal distributed LM aggregate checkpoint drifted from generator output")
if promotion != generated_promotion:
    fail("A1 minimal distributed LM promotion receipt drifted from generator output")

if report["lane_id"] != "a1_minimal_distributed_lm_001":
    fail("trusted aggregation report lost lane id")
if report["trusted_aggregation_rule"] != "trusted_weighted_delta_average_v1":
    fail("trusted aggregation report lost aggregation rule")
if report["accepted_contribution_count"] < 2:
    fail("trusted aggregation must consume at least two accepted updates")
if report["model_progress_participant_count"] != report["accepted_contribution_count"]:
    fail("model-progress participant count must match accepted aggregate inputs")
if report["aggregation_weight_basis"] != "tokens":
    fail("trusted aggregation weight basis drifted")
if report["total_aggregation_weight"] != sum(update["aggregation_weight_value"] for update in report["accepted_local_updates"]):
    fail("trusted aggregation total weight drifted")
if len({update["contribution_id"] for update in report["accepted_local_updates"]}) != len(report["accepted_local_updates"]):
    fail("trusted aggregation contribution ids must be distinct")
for update in report["accepted_local_updates"]:
    if update["run_id"] != report["run_id"]:
        fail("accepted update run id drifted")
    if update["window_id"] != report["aggregate_window_id"]:
        fail("accepted update window id drifted")
    if update["validator_disposition"] != "accepted":
        fail("trusted aggregation may consume only accepted updates")
    if not update["accepted_for_aggregation"]:
        fail("accepted update must be accepted_for_aggregation")
    if update["aggregation_weight_basis"] != "tokens":
        fail("accepted update weight basis drifted")

if checkpoint["accepted_aggregate_id"] != report["accepted_aggregate_id"]:
    fail("aggregate checkpoint accepted aggregate id drifted")
if checkpoint["aggregated_delta_digest"] != report["aggregated_delta_digest"]:
    fail("aggregate checkpoint delta digest drifted")
if checkpoint["checkpoint_digest"] != report["output_checkpoint_digest"]:
    fail("aggregate checkpoint digest drifted from report output")
if checkpoint["promoted_checkpoint_ref"] != report["promoted_checkpoint_ref"]:
    fail("aggregate checkpoint promoted ref drifted")

if promotion["accepted_aggregate_id"] != report["accepted_aggregate_id"]:
    fail("promotion receipt accepted aggregate id drifted")
if promotion["aggregated_delta_digest"] != report["aggregated_delta_digest"]:
    fail("promotion receipt delta digest drifted")
if promotion["output_checkpoint_pointer"] != report["output_checkpoint_pointer"]:
    fail("promotion receipt output pointer drifted")
if promotion["promoted_checkpoint_ref"] != report["promoted_checkpoint_ref"]:
    fail("promotion receipt promoted checkpoint ref drifted")
if promotion["promotion_verdict"] != "promoted":
    fail("promotion receipt must be promoted")
if promotion["validation_loss_after"] > promotion["validation_loss_before"]:
    fail("promotion worsened validation loss")

summary = {
    "verdict": "verified",
    "lane_id": report["lane_id"],
    "accepted_contribution_count": report["accepted_contribution_count"],
    "model_progress_participant_count": report["model_progress_participant_count"],
    "total_aggregation_weight": report["total_aggregation_weight"],
    "aggregated_delta_digest": report["aggregated_delta_digest"],
    "output_checkpoint_digest": report["output_checkpoint_digest"],
    "promoted_checkpoint_ref": report["promoted_checkpoint_ref"],
    "validation_loss_before": report["validation_loss_before"],
    "validation_loss_after": report["validation_loss_after"],
    "promotion_receipt_digest": promotion["promotion_receipt_digest"],
    "aggregate_report_digest": report["aggregate_report_digest"],
}
print(json.dumps(summary, indent=2))
PY
