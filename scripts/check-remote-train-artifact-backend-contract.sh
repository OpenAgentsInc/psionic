#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/remote_train_artifact_backend_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/remote_train_artifact_backend_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/remote_train_artifact_backend_contract_v1.json"
cargo run -q -p psionic-train --bin remote_train_artifact_backend_contract -- "${generated_path}" >/dev/null

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
    fail("remote train artifact backend check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.remote_train_artifact_backend_contract.v1":
    fail("remote train artifact backend check: schema version drifted")

backend_ids = {backend["backend_id"] for backend in fixture["backends"]}
required_backends = {"google_train_bucket_backend", "runpod_workspace_backend"}
if backend_ids != required_backends:
    fail("remote train artifact backend check: backend set drifted")

placement_classes = {decision["artifact_class"] for decision in fixture["placement_decisions"]}
required_classes = {
    "checkpoint",
    "log_bundle",
    "metrics_bundle",
    "final_evidence_bundle",
}
if placement_classes != required_classes:
    fail("remote train artifact backend check: placement coverage drifted")

projection_pairs = {
    (projection["source_id"], projection["artifact_class"])
    for projection in fixture["finalizer_projections"]
}
for source_id in ("google_l4_validator_node", "runpod_8xh100_dense_node"):
    for artifact_class in required_classes:
        if (source_id, artifact_class) not in projection_pairs:
            fail("remote train artifact backend check: finalizer projection coverage drifted")

summary = {
    "verdict": "verified",
    "backend_count": len(fixture["backends"]),
    "placement_class_count": len(placement_classes),
    "finalizer_projection_count": len(fixture["finalizer_projections"]),
}
print(json.dumps(summary, indent=2))
PY
