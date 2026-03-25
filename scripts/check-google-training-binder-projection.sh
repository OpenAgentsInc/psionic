#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/google_training_binder_projection_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/google_training_binder_projection.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/google_training_binder_projection_v1.json"
cargo run -q -p psionic-train --bin google_training_binder_projection -- "${generated_path}" >/dev/null

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
    fail("google training binder projection check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.google_training_binder_projection.v1":
    fail("google training binder projection check: schema version drifted")
if len(fixture["lane_projections"]) != 2:
    fail("google training binder projection check: lane count drifted")

lane_kinds = {lane["lane_kind"] for lane in fixture["lane_projections"]}
if lane_kinds != {"single_node_accelerated", "two_node_configured_peer_swarm"}:
    fail(f"google training binder projection check: lane kinds drifted: {sorted(lane_kinds)}")

for lane in fixture["lane_projections"]:
    if not lane["retained_checker_paths"]:
        fail(f"google training binder projection check: lane {lane['lane_projection_id']} lost checker coverage")
    if not lane["retained_evidence_paths"]:
        fail(f"google training binder projection check: lane {lane['lane_projection_id']} lost retained evidence paths")
    if not lane["runbook_path"].startswith("docs/PSION_GOOGLE_"):
        fail(f"google training binder projection check: lane {lane['lane_projection_id']} runbook path drifted")

summary = {
    "verdict": "verified",
    "lane_count": len(fixture["lane_projections"]),
    "lane_kinds": sorted(lane_kinds),
}
print(json.dumps(summary, indent=2))
PY
