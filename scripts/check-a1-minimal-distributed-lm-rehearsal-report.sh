#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_dir="${repo_root}/fixtures/psion/a1_minimal_distributed_lm"
rehearsal_path="${fixture_dir}/operator_rehearsal_report_v1.json"
local_report_path="${fixture_dir}/local_update_report_v1.json"
local_receipt_path="${fixture_dir}/local_update_contribution_receipt_v1.json"
aggregation_report_path="${fixture_dir}/trusted_aggregation_report_v1.json"
promotion_path="${fixture_dir}/promotion_receipt_v1.json"

python3 - "${rehearsal_path}" "${local_report_path}" "${local_receipt_path}" "${aggregation_report_path}" "${promotion_path}" <<'PY'
import json
import sys
from pathlib import Path

rehearsal_path = Path(sys.argv[1])
local_report_path = Path(sys.argv[2])
local_receipt_path = Path(sys.argv[3])
aggregation_report_path = Path(sys.argv[4])
promotion_path = Path(sys.argv[5])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
local_report = json.loads(local_report_path.read_text(encoding="utf-8"))
local_receipt = json.loads(local_receipt_path.read_text(encoding="utf-8"))
aggregation = json.loads(aggregation_report_path.read_text(encoding="utf-8"))
promotion = json.loads(promotion_path.read_text(encoding="utf-8"))

if rehearsal["schema_version"] != "psion.a1_minimal_distributed_lm.operator_rehearsal_report.v1":
    fail("rehearsal report schema drifted")
if rehearsal["lane_id"] != "a1_minimal_distributed_lm_001":
    fail("rehearsal report lane id drifted")
if "scripts/check-a1-minimal-distributed-lm-rehearsal-report.sh" not in rehearsal["verification_commands"]:
    fail("rehearsal report must list its checker")

single = rehearsal["single_node_rehearsal"]
if single["status"] != "retained_verified":
    fail("single-node rehearsal must be retained_verified")
if single["run_id"] != local_report["run_id"]:
    fail("single-node run id drifted from local-update report")
if single["assignment_id"] != local_report["assignment_id"]:
    fail("single-node assignment id drifted from local-update report")
if single["worker_id"] != local_report["worker_id"]:
    fail("single-node worker id drifted from local-update report")
if single["local_step_count"] != local_report["local_step_count"]:
    fail("single-node local step count drifted")
if single["consumed_token_count"] != local_report["consumed_token_count"]:
    fail("single-node consumed token count drifted")
if single["training_loss_before"] != local_report["loss_before"]:
    fail("single-node loss_before drifted")
if single["training_loss_after"] != local_report["loss_after"]:
    fail("single-node loss_after drifted")
if single["validation_loss_before"] != local_report["validation_loss_before"]:
    fail("single-node validation_loss_before drifted")
if single["validation_loss_after"] != local_report["validation_loss_after"]:
    fail("single-node validation_loss_after drifted")
if single["checkpoint_step4_digest"] != local_report["checkpoint_step4"]["checkpoint_digest"]:
    fail("single-node checkpoint digest drifted")
if single["delta_digest"] != local_report["delta_digest"]:
    fail("single-node delta digest drifted")
if single["report_digest"] != local_report["report_digest"]:
    fail("single-node report digest drifted")
if single["validator_disposition"] != local_receipt["validator_disposition"]:
    fail("single-node validator disposition drifted")
if single["accepted_for_aggregation"] != local_receipt["accepted_for_aggregation"]:
    fail("single-node accepted_for_aggregation drifted")
if single["accepted_for_aggregation"] or single["model_progress_participant_count"] != 0:
    fail("single-node fixture must not pre-claim accepted model-progress credit")

trusted = rehearsal["trusted_aggregation_rehearsal"]
if trusted["status"] != "retained_verified":
    fail("trusted aggregation rehearsal must be retained_verified")
if trusted["physical_multi_host_status"] != "not_produced_from_this_session":
    fail("trusted aggregation report must keep the physical multi-host gap explicit")
if trusted["run_id"] != aggregation["run_id"]:
    fail("trusted aggregation run id drifted")
if trusted["accepted_contribution_count"] != aggregation["accepted_contribution_count"]:
    fail("accepted contribution count drifted")
if trusted["model_progress_participant_count"] != aggregation["model_progress_participant_count"]:
    fail("model-progress participant count drifted")
if trusted["total_aggregation_weight"] != aggregation["total_aggregation_weight"]:
    fail("aggregation weight drifted")
if trusted["validation_loss_before"] != aggregation["validation_loss_before"]:
    fail("aggregation validation_loss_before drifted")
if trusted["validation_loss_after"] != aggregation["validation_loss_after"]:
    fail("aggregation validation_loss_after drifted")
if trusted["aggregated_delta_digest"] != aggregation["aggregated_delta_digest"]:
    fail("aggregated delta digest drifted")
if trusted["output_checkpoint_digest"] != aggregation["output_checkpoint_digest"]:
    fail("output checkpoint digest drifted")
if trusted["promoted_checkpoint_ref"] != aggregation["promoted_checkpoint_ref"]:
    fail("promoted checkpoint ref drifted")
if trusted["promotion_receipt_digest"] != promotion["promotion_receipt_digest"]:
    fail("promotion receipt digest drifted")
if trusted["aggregate_report_digest"] != aggregation["aggregate_report_digest"]:
    fail("aggregate report digest drifted")
if promotion["promotion_verdict"] != "promoted":
    fail("promotion receipt must remain promoted")
if len(trusted["host_roles"]) < 3:
    fail("trusted rehearsal must retain two worker roles plus an aggregation role")

worker_roles = [
    role for role in trusted["host_roles"]
    if role["role"].startswith("local_update_worker_")
]
if len(worker_roles) != len(aggregation["accepted_local_updates"]):
    fail("trusted rehearsal worker role count must match accepted local updates")

for role, update in zip(worker_roles, aggregation["accepted_local_updates"]):
    for key in [
        "worker_id",
        "node_pubkey",
        "assignment_id",
        "contribution_id",
        "consumed_token_count",
        "output_checkpoint_digest",
        "output_delta_digest",
        "source_contribution_digest",
        "accepted_for_aggregation",
    ]:
        if role[key] != update[key]:
            fail(f"trusted rehearsal worker role {role['role']} drifted on {key}")
    if not role["accepted_for_aggregation"]:
        fail(f"trusted rehearsal worker role {role['role']} must be accepted for aggregation")

if rehearsal["physical_host_access_probe"]["status"] != "blocked_from_this_session":
    fail("physical host access probe must remain explicit until real host receipts are retained")
if "does not prove a live physical tri-host rehearsal" not in rehearsal["claim_boundary"]:
    fail("rehearsal claim boundary must keep physical tri-host limitation explicit")

summary = {
    "verdict": "verified",
    "lane_id": rehearsal["lane_id"],
    "single_node_run_id": single["run_id"],
    "single_node_checkpoint_digest": single["checkpoint_step4_digest"],
    "trusted_aggregation_run_id": trusted["run_id"],
    "accepted_contribution_count": trusted["accepted_contribution_count"],
    "model_progress_participant_count": trusted["model_progress_participant_count"],
    "promoted_checkpoint_ref": trusted["promoted_checkpoint_ref"],
    "physical_multi_host_status": trusted["physical_multi_host_status"],
}
print(json.dumps(summary, indent=2))
PY
