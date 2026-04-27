#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_dir="${repo_root}/fixtures/psion/a1_minimal_distributed_lm"
report_path="${fixture_dir}/local_update_report_v1.json"
checkpoint_step2_path="${fixture_dir}/local_update_checkpoint_step2_v1.json"
checkpoint_step4_path="${fixture_dir}/local_update_checkpoint_step4_v1.json"
artifact_manifest_path="${fixture_dir}/local_update_artifact_manifest_v1.json"
contribution_receipt_path="${fixture_dir}/local_update_contribution_receipt_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/a1_minimal_distributed_lm_local_update.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

cargo run -q -p psionic-train --example a1_minimal_distributed_lm_local_update_fixture -- "${tmpdir}" >/dev/null

generated_dir="${tmpdir}/fixtures/psion/a1_minimal_distributed_lm"
generated_report_path="${generated_dir}/local_update_report_v1.json"
generated_checkpoint_step2_path="${generated_dir}/local_update_checkpoint_step2_v1.json"
generated_checkpoint_step4_path="${generated_dir}/local_update_checkpoint_step4_v1.json"
generated_artifact_manifest_path="${generated_dir}/local_update_artifact_manifest_v1.json"
generated_contribution_receipt_path="${generated_dir}/local_update_contribution_receipt_v1.json"

python3 - "${report_path}" "${generated_report_path}" "${checkpoint_step2_path}" "${generated_checkpoint_step2_path}" "${checkpoint_step4_path}" "${generated_checkpoint_step4_path}" "${artifact_manifest_path}" "${generated_artifact_manifest_path}" "${contribution_receipt_path}" "${generated_contribution_receipt_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
generated_report_path = Path(sys.argv[2])
checkpoint_step2_path = Path(sys.argv[3])
generated_checkpoint_step2_path = Path(sys.argv[4])
checkpoint_step4_path = Path(sys.argv[5])
generated_checkpoint_step4_path = Path(sys.argv[6])
artifact_manifest_path = Path(sys.argv[7])
generated_artifact_manifest_path = Path(sys.argv[8])
contribution_receipt_path = Path(sys.argv[9])
generated_contribution_receipt_path = Path(sys.argv[10])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

report = json.loads(report_path.read_text(encoding="utf-8"))
generated_report = json.loads(generated_report_path.read_text(encoding="utf-8"))
checkpoint_step2 = json.loads(checkpoint_step2_path.read_text(encoding="utf-8"))
generated_checkpoint_step2 = json.loads(generated_checkpoint_step2_path.read_text(encoding="utf-8"))
checkpoint_step4 = json.loads(checkpoint_step4_path.read_text(encoding="utf-8"))
generated_checkpoint_step4 = json.loads(generated_checkpoint_step4_path.read_text(encoding="utf-8"))
artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
generated_artifact_manifest = json.loads(generated_artifact_manifest_path.read_text(encoding="utf-8"))
contribution_receipt = json.loads(contribution_receipt_path.read_text(encoding="utf-8"))
generated_contribution_receipt = json.loads(generated_contribution_receipt_path.read_text(encoding="utf-8"))

if report != generated_report:
    fail("A1 minimal distributed LM local-update report drifted from generator output")
if checkpoint_step2 != generated_checkpoint_step2:
    fail("A1 minimal distributed LM local-update step2 checkpoint drifted from generator output")
if checkpoint_step4 != generated_checkpoint_step4:
    fail("A1 minimal distributed LM local-update step4 checkpoint drifted from generator output")
if artifact_manifest != generated_artifact_manifest:
    fail("A1 minimal distributed LM local-update artifact manifest drifted from generator output")
if contribution_receipt != generated_contribution_receipt:
    fail("A1 minimal distributed LM local-update contribution receipt drifted from generator output")

if report["lane_id"] != "a1_minimal_distributed_lm_001":
    fail("local-update report lost the lane id")
if report["finite_difference_used"]:
    fail("local-update report must not use finite-difference gradients")
if report["backward_path_kind"] != "analytic_lm_head_cross_entropy_backward_v1":
    fail("local-update report lost the analytic backward kind")
if report["trained_parameter_paths"] != ["lm_head.weight"]:
    fail("first production proof must stay LM-head-only")
if report["loss_before"] == report["loss_after"]:
    fail("training loss did not change")
if report["validation_loss_before"] == report["validation_loss_after"]:
    fail("validation loss did not change")
if not report["resume_matches_uninterrupted"]:
    fail("resumed local update does not match uninterrupted local update")
if report["resumed_final_model_state_digest"] != report["uninterrupted_final_model_state_digest"]:
    fail("resumed model digest does not match uninterrupted model digest")
if report["resumed_final_optimizer_state_digest"] != report["uninterrupted_final_optimizer_state_digest"]:
    fail("resumed optimizer digest does not match uninterrupted optimizer digest")

if artifact_manifest["run_id"] != report["run_id"]:
    fail("artifact manifest run_id drifted from report")
if artifact_manifest["assignment_id"] != report["assignment_id"]:
    fail("artifact manifest assignment_id drifted from report")
if artifact_manifest["worker_id"] != report["worker_id"]:
    fail("artifact manifest worker_id drifted from report")
if artifact_manifest["work_class"] != "local_update_training":
    fail("artifact manifest lost local_update_training work class")
if artifact_manifest["output_checkpoint_digest"] != report["checkpoint_step4"]["checkpoint_digest"]:
    fail("artifact manifest output checkpoint digest drifted from report")
if artifact_manifest["output_delta_digest"] != report["delta_digest"]:
    fail("artifact manifest output delta digest drifted from report")
if artifact_manifest["artifact_count"] != len(artifact_manifest["artifacts"]):
    fail("artifact manifest artifact count drifted")
if not all(not path["path"].startswith("/") for path in artifact_manifest["materialized_paths"]):
    fail("committed artifact manifest must not carry absolute materialized paths")

if contribution_receipt["run_id"] != report["run_id"]:
    fail("contribution receipt run_id drifted from report")
if contribution_receipt["training_run_id"] != report["run_id"]:
    fail("contribution receipt training_run_id drifted from report")
if contribution_receipt["assignment_id"] != report["assignment_id"]:
    fail("contribution receipt assignment_id drifted from report")
if contribution_receipt["worker_id"] != report["worker_id"]:
    fail("contribution receipt worker_id drifted from report")
if contribution_receipt["tokenizer_digest"] != report["tokenizer_digest"]:
    fail("contribution receipt tokenizer digest drifted from report")
if contribution_receipt["tokenized_dataset_digest"] != report["tokenized_dataset_digest"]:
    fail("contribution receipt tokenized dataset digest drifted from report")
if contribution_receipt["base_checkpoint_ref"] != report["base_checkpoint_ref"]:
    fail("contribution receipt base checkpoint ref drifted from report")
if contribution_receipt["local_step_count"] != report["local_step_count"]:
    fail("contribution receipt local_step_count drifted from report")
if contribution_receipt["consumed_token_count"] != report["consumed_token_count"]:
    fail("contribution receipt consumed_token_count drifted from report")
if contribution_receipt["output_checkpoint_digest"] != report["checkpoint_step4"]["checkpoint_digest"]:
    fail("contribution receipt output checkpoint digest drifted from report")
if contribution_receipt["output_delta_digest"] != report["delta_digest"]:
    fail("contribution receipt output delta digest drifted from report")
if contribution_receipt["object_digest"] != report["checkpoint_step4"]["checkpoint_digest"]:
    fail("contribution receipt object digest must bind output checkpoint digest")
if contribution_receipt["manifest_digest"] != artifact_manifest["artifact_manifest_digest"]:
    fail("contribution receipt manifest_digest drifted from artifact manifest")
if contribution_receipt["artifact_manifest_digest"] != artifact_manifest["artifact_manifest_digest"]:
    fail("contribution receipt artifact_manifest_digest drifted from artifact manifest")
if contribution_receipt["validator_disposition"] != "replay_required":
    fail("fixture contribution receipt must stay validator replay required")
if contribution_receipt["validator_verdict_binding"] != "pending_validator_replay":
    fail("fixture contribution receipt must bind pending validator replay")
if contribution_receipt["aggregation_eligibility"] != "eligible":
    fail("contribution receipt must remain aggregation eligible after acceptance")
if contribution_receipt["accepted_for_aggregation"]:
    fail("fixture contribution receipt must not pre-claim Nexus aggregation acceptance")
if contribution_receipt["aggregation_weight_basis"] != "tokens":
    fail("contribution receipt aggregation weight basis drifted")
if contribution_receipt["aggregation_weight_value"] != report["consumed_token_count"]:
    fail("contribution receipt aggregation weight value must equal consumed tokens")
if contribution_receipt["aggregation_weight_bps"] != 10000:
    fail("contribution receipt aggregation weight bps drifted")
if not contribution_receipt["model_progress_eligible"]:
    fail("local-update receipt must remain model-progress eligible")
if contribution_receipt["closeout_verdict_binding"] != "pending_nexus_closeout":
    fail("fixture contribution receipt must bind pending Nexus closeout")

summary = {
    "verdict": "verified",
    "lane_id": report["lane_id"],
    "backward_path_kind": report["backward_path_kind"],
    "finite_difference_used": report["finite_difference_used"],
    "local_step_count": report["local_step_count"],
    "consumed_token_count": report["consumed_token_count"],
    "loss_before": report["loss_before"],
    "loss_after": report["loss_after"],
    "validation_loss_before": report["validation_loss_before"],
    "validation_loss_after": report["validation_loss_after"],
    "checkpoint_step4_digest": report["checkpoint_step4"]["checkpoint_digest"],
    "delta_digest": report["delta_digest"],
    "report_digest": report["report_digest"],
    "artifact_manifest_digest": artifact_manifest["artifact_manifest_digest"],
    "contribution_digest": contribution_receipt["contribution_digest"],
    "validator_disposition": contribution_receipt["validator_disposition"],
    "accepted_for_aggregation": contribution_receipt["accepted_for_aggregation"],
    "aggregation_weight_value": contribution_receipt["aggregation_weight_value"],
}
print(json.dumps(summary, indent=2))
PY
