#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/first_same_job_mixed_backend_dense_run_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/first_same_job_mixed_backend_dense_run.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/first_same_job_mixed_backend_dense_run_v1.json"
cargo run -q -p psionic-train --bin first_same_job_mixed_backend_dense_run -- "${generated_path}" >/dev/null

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
    fail("same-job mixed-backend dense run check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.first_same_job_mixed_backend_dense_run.v1":
    fail("same-job mixed-backend dense run check: schema version drifted")

if fixture["world_size"] != 9:
    fail("same-job mixed-backend dense run check: world size drifted")
if len(fixture["participants"]) != 2:
    fail("same-job mixed-backend dense run check: participant count drifted")
if len(fixture["step_metrics"]) != 4:
    fail("same-job mixed-backend dense run check: step metric count drifted")
if len(fixture["checkpoint_events"]) != 1:
    fail("same-job mixed-backend dense run check: checkpoint event count drifted")

families = sorted(participant["backend_family"] for participant in fixture["participants"])
if families != ["cuda", "mlx_metal"]:
    fail("same-job mixed-backend dense run check: backend families drifted")

if fixture["final_disposition"] != "bounded_success":
    fail("same-job mixed-backend dense run check: final disposition drifted")

summary = {
    "verdict": "verified",
    "run_id": fixture["run_id"],
    "world_size": fixture["world_size"],
    "participant_count": len(fixture["participants"]),
    "checkpoint_events": len(fixture["checkpoint_events"]),
    "final_disposition": fixture["final_disposition"],
}
print(json.dumps(summary, indent=2))
PY
