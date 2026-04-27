#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/a1_minimal_distributed_lm_lane_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/a1_minimal_distributed_lm_lane_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/a1_minimal_distributed_lm_lane_contract_v1.json"
cargo run -q -p psionic-train --example a1_minimal_distributed_lm_lane_contract_fixture -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

committed = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

if committed != generated:
    fail("A1 minimal distributed LM lane contract fixture drifted from generator output")

if committed["lane_id"] != "a1_minimal_distributed_lm_001":
    fail("A1 minimal distributed LM lane contract lost its lane id")
if committed["lane_id"] == "psion_cs336_a1_demo_v1":
    fail("A1 minimal distributed LM lane contract reused the bounded demo lane id")
if committed["closeout_and_promotion"]["participant_counter_source"] != "training_accepted_contributors":
    fail("participant counter source drifted")
if committed["closeout_and_promotion"]["model_progress_counter_source"] != "training_model_progress_contributors":
    fail("model-progress counter source drifted")

required_fields = set(committed["contribution_receipt_schema"]["required_fields"])
expected_fields = {
    "worker_id",
    "assignment_id",
    "run_id",
    "input_shard",
    "base_checkpoint",
    "output_artifact",
    "loss_before",
    "loss_after",
    "validator_verdict",
}
if not expected_fields.issubset(required_fields):
    fail("contribution receipt schema lost required OpenAgents fields")

support_classes = set(committed["contribution_receipt_schema"]["support_work_classes"])
expected_support_classes = {
    "tokenized_shard_validation",
    "validation_replay",
    "checkpoint_verification",
    "eval_batch",
    "artifact_rematerialization",
    "independent_scored_training_window",
}
if support_classes != expected_support_classes:
    fail(f"support work class set drifted: {sorted(support_classes)}")

summary = {
    "verdict": "verified",
    "lane_id": committed["lane_id"],
    "release_id": committed["release_id"],
    "environment_ref": committed["environment_ref"],
    "contract_digest": committed["contract_digest"],
}
print(json.dumps(summary, indent=2))
PY
