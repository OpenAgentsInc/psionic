#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_dir="${repo_root}/fixtures/psion/a1_minimal_distributed_lm"
report_path="${fixture_dir}/local_update_report_v1.json"
checkpoint_step2_path="${fixture_dir}/local_update_checkpoint_step2_v1.json"
checkpoint_step4_path="${fixture_dir}/local_update_checkpoint_step4_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/a1_minimal_distributed_lm_local_update.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

cargo run -q -p psionic-train --example a1_minimal_distributed_lm_local_update_fixture -- "${tmpdir}" >/dev/null

generated_dir="${tmpdir}/fixtures/psion/a1_minimal_distributed_lm"
generated_report_path="${generated_dir}/local_update_report_v1.json"
generated_checkpoint_step2_path="${generated_dir}/local_update_checkpoint_step2_v1.json"
generated_checkpoint_step4_path="${generated_dir}/local_update_checkpoint_step4_v1.json"

python3 - "${report_path}" "${generated_report_path}" "${checkpoint_step2_path}" "${generated_checkpoint_step2_path}" "${checkpoint_step4_path}" "${generated_checkpoint_step4_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
generated_report_path = Path(sys.argv[2])
checkpoint_step2_path = Path(sys.argv[3])
generated_checkpoint_step2_path = Path(sys.argv[4])
checkpoint_step4_path = Path(sys.argv[5])
generated_checkpoint_step4_path = Path(sys.argv[6])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

report = json.loads(report_path.read_text(encoding="utf-8"))
generated_report = json.loads(generated_report_path.read_text(encoding="utf-8"))
checkpoint_step2 = json.loads(checkpoint_step2_path.read_text(encoding="utf-8"))
generated_checkpoint_step2 = json.loads(generated_checkpoint_step2_path.read_text(encoding="utf-8"))
checkpoint_step4 = json.loads(checkpoint_step4_path.read_text(encoding="utf-8"))
generated_checkpoint_step4 = json.loads(generated_checkpoint_step4_path.read_text(encoding="utf-8"))

if report != generated_report:
    fail("A1 minimal distributed LM local-update report drifted from generator output")
if checkpoint_step2 != generated_checkpoint_step2:
    fail("A1 minimal distributed LM local-update step2 checkpoint drifted from generator output")
if checkpoint_step4 != generated_checkpoint_step4:
    fail("A1 minimal distributed LM local-update step4 checkpoint drifted from generator output")

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
}
print(json.dumps(summary, indent=2))
PY
