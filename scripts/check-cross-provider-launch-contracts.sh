#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_dir="${repo_root}/fixtures/training/launch_contracts"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/cross_provider_launch_contracts.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_dir="${tmpdir}/launch_contracts"
cargo run -q -p psionic-train --bin cross_provider_launch_contracts -- "${generated_dir}" >/dev/null

python3 - "${fixture_dir}" "${generated_dir}" <<'PY'
import json
import sys
from pathlib import Path

fixture_dir = Path(sys.argv[1])
generated_dir = Path(sys.argv[2])
expected_files = {
    "google_single_node_accelerated_v1.json",
    "google_two_node_swarm_v1.json",
    "runpod_8xh100_v1.json",
    "local_first_swarm_v1.json",
}

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

fixture_files = {path.name for path in fixture_dir.iterdir() if path.is_file()}
generated_files = {path.name for path in generated_dir.iterdir() if path.is_file()}
if fixture_files != expected_files:
    fail(f"launch-contract check: fixture dir file set drifted: {sorted(fixture_files)}")
if generated_files != expected_files:
    fail(f"launch-contract check: generated dir file set drifted: {sorted(generated_files)}")

for filename in sorted(expected_files):
    committed = json.loads((fixture_dir / filename).read_text(encoding="utf-8"))
    generated = json.loads((generated_dir / filename).read_text(encoding="utf-8"))
    if committed != generated:
        fail(f"launch-contract check: {filename} drifted from generator output")

google_single = json.loads((fixture_dir / "google_single_node_accelerated_v1.json").read_text(encoding="utf-8"))
google_swarm = json.loads((fixture_dir / "google_two_node_swarm_v1.json").read_text(encoding="utf-8"))
runpod = json.loads((fixture_dir / "runpod_8xh100_v1.json").read_text(encoding="utf-8"))
local_swarm = json.loads((fixture_dir / "local_first_swarm_v1.json").read_text(encoding="utf-8"))

if google_single["requested_execution_class"] != "dense_full_model_rank":
    fail("launch-contract check: google single-node execution class drifted")
if google_single["startup_plan"]["startup_entrypoint"] != "scripts/psion-google-single-node-startup.sh":
    fail("launch-contract check: google single-node startup entrypoint drifted")
if google_swarm["cluster_port_bindings"][0]["port"] != 34100:
    fail("launch-contract check: google swarm cluster port drifted")
if runpod["finalizer_plan"]["finalizer_entrypoint"] != "scripts/parameter-golf-runpod-finalize-8xh100.sh":
    fail("launch-contract check: runpod finalizer entrypoint drifted")
if "training_visualization/remote_training_run_index_v1.json" not in runpod["finalizer_plan"]["expected_output_artifacts"]:
    fail("launch-contract check: runpod visualization output drifted")
if local_swarm["projected_steps"][0]["argv_template"][-1] != "--manifest-only":
    fail("launch-contract check: local swarm manifest-only projection drifted")

required_envs = {
    "PSION_PROGRAM_MANIFEST_ID",
    "PSION_PROGRAM_MANIFEST_DIGEST",
    "PSION_RUN_ID",
    "PSION_EXECUTION_CLASS",
    "PSION_RUN_ROOT",
    "PSION_LAUNCH_ROOT",
    "PSION_CHECKPOINT_ROOT",
    "PSION_METRICS_ROOT",
    "PSION_VISUALIZATION_ROOT",
    "PSION_FINAL_ROOT",
}
for name, contract in {
    "google_single": google_single,
    "google_swarm": google_swarm,
    "runpod": runpod,
    "local_swarm": local_swarm,
}.items():
    env_names = {item["name"] for item in contract["runtime_env"]}
    if not required_envs.issubset(env_names):
        fail(f"launch-contract check: {name} lost one or more required shared env vars")

summary = {
    "verdict": "verified",
    "contract_count": len(expected_files),
    "binders": [
        google_single["binder_kind"],
        google_swarm["binder_kind"],
        runpod["binder_kind"],
        local_swarm["binder_kind"],
    ],
}
print(json.dumps(summary, indent=2))
PY
