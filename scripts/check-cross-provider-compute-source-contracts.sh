#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_dir="${repo_root}/fixtures/training/compute_sources"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/cross_provider_compute_sources.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_dir="${tmpdir}/compute_sources"
cargo run -q -p psionic-train --bin cross_provider_compute_source_contracts -- "${generated_dir}" >/dev/null

python3 - "${fixture_dir}" "${generated_dir}" <<'PY'
import json
import sys
from pathlib import Path

fixture_dir = Path(sys.argv[1])
generated_dir = Path(sys.argv[2])
expected_files = {
    "google_l4_validator_node_v1.json",
    "runpod_8xh100_dense_node_v1.json",
    "local_rtx4080_workstation_v1.json",
    "local_mlx_mac_workstation_v1.json",
    "planner_input_v1.json",
    "launch_inputs_v1.json",
}

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

fixture_files = {path.name for path in fixture_dir.iterdir() if path.is_file()}
generated_files = {path.name for path in generated_dir.iterdir() if path.is_file()}
if fixture_files != expected_files:
    fail(f"compute-source contract check: fixture dir file set drifted: {sorted(fixture_files)}")
if generated_files != expected_files:
    fail(f"compute-source contract check: generated dir file set drifted: {sorted(generated_files)}")

for filename in sorted(expected_files):
    committed = json.loads((fixture_dir / filename).read_text(encoding="utf-8"))
    generated = json.loads((generated_dir / filename).read_text(encoding="utf-8"))
    if committed != generated:
        fail(f"compute-source contract check: {filename} drifted from generator output")

google = json.loads((fixture_dir / "google_l4_validator_node_v1.json").read_text(encoding="utf-8"))
runpod = json.loads((fixture_dir / "runpod_8xh100_dense_node_v1.json").read_text(encoding="utf-8"))
local_rtx = json.loads((fixture_dir / "local_rtx4080_workstation_v1.json").read_text(encoding="utf-8"))
local_mac = json.loads((fixture_dir / "local_mlx_mac_workstation_v1.json").read_text(encoding="utf-8"))
planner = json.loads((fixture_dir / "planner_input_v1.json").read_text(encoding="utf-8"))
launch_inputs = json.loads((fixture_dir / "launch_inputs_v1.json").read_text(encoding="utf-8"))

if google["provider"] != "google_cloud":
    fail("compute-source contract check: google provider drifted")
if runpod["admitted_execution_classes"] != [
    "dense_full_model_rank",
    "checkpoint_writer",
    "eval_worker",
    "data_builder",
]:
    fail("compute-source contract check: runpod admitted execution classes drifted")
if "validated_contributor_window" not in local_rtx["admitted_execution_classes"]:
    fail("compute-source contract check: local RTX source lost validated contributor admission")
if "validator" not in local_mac["admitted_execution_classes"]:
    fail("compute-source contract check: local Mac source lost validator admission")

refused = [
    candidate for candidate in planner["candidates"]
    if candidate["source_id"] == "local_mlx_mac_workstation"
    and candidate["requested_execution_class"] == "dense_full_model_rank"
]
if len(refused) != 1 or refused[0]["expected_disposition"] != "refused":
    fail("compute-source contract check: planner refusal case drifted")

if len(launch_inputs) != 4:
    fail("compute-source contract check: launch input count drifted")
if launch_inputs[1]["source_id"] != "runpod_8xh100_dense_node":
    fail("compute-source contract check: runpod launch input ordering drifted")

summary = {
    "verdict": "verified",
    "fixture_count": len(expected_files),
    "sources": [
        google["source_id"],
        runpod["source_id"],
        local_rtx["source_id"],
        local_mac["source_id"],
    ],
    "planner_candidates": len(planner["candidates"]),
    "launch_inputs": len(launch_inputs),
}
print(json.dumps(summary, indent=2))
PY
