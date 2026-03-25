#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/cross_provider_admission_plan_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/cross_provider_admission_plan.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/cross_provider_admission_plan_v1.json"
cargo run -q -p psionic-train --bin cross_provider_admission_plan -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if fixture != generated:
    fail("cross-provider admission planner check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.cross_provider_admission_planner.v1":
    fail("cross-provider admission planner check: schema version drifted")

def selected_for(role: str):
    return [
        item for item in fixture["candidate_evaluations"]
        if item["requested_execution_class"] == role and item["selected"]
    ]

dense = selected_for("dense_full_model_rank")
if len(dense) != 1 or dense[0]["source_id"] != "runpod_8xh100_dense_node":
    fail("cross-provider admission planner check: dense-rank placement drifted")

checkpoint = selected_for("checkpoint_writer")
if len(checkpoint) != 1 or checkpoint[0]["source_id"] != "google_l4_validator_node":
    fail("cross-provider admission planner check: checkpoint-writer placement drifted")

validators = selected_for("validator")
validator_ids = {item["source_id"] for item in validators}
if validator_ids != {"google_l4_validator_node", "local_mlx_mac_workstation"}:
    fail(f"cross-provider admission planner check: validator set drifted: {sorted(validator_ids)}")

data_builder_refusals = [
    item for item in fixture["candidate_evaluations"]
    if item["requested_execution_class"] == "data_builder"
    and item["source_id"] == "runpod_8xh100_dense_node"
]
if not data_builder_refusals or data_builder_refusals[0]["refusal_kind"] != "cost_posture_rejected":
    fail("cross-provider admission planner check: runpod data-builder refusal drifted")

summary = {
    "verdict": "verified",
    "evaluation_count": len(fixture["candidate_evaluations"]),
    "selected_dense_rank_source": dense[0]["source_id"],
    "selected_checkpoint_writer_source": checkpoint[0]["source_id"],
}
print(json.dumps(summary, indent=2))
PY
