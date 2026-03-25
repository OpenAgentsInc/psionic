#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/provider_neutral_training_execution_evidence_bundle_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/training_execution_evidence_bundle.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/provider_neutral_training_execution_evidence_bundle_v1.json"
cargo run -q -p psionic-train --bin training_execution_evidence_bundle -- "${generated_path}" >/dev/null

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
    fail("training execution evidence bundle check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.training_execution_evidence_bundle.v1":
    fail("training execution evidence bundle check: schema version drifted")

topology_kinds = {segment["topology_kind"] for segment in fixture["segment_evidence"]}
required_topologies = {
    "single_node",
    "dense_distributed",
    "contributor_window",
    "validator_only",
    "hybrid",
}
if topology_kinds != required_topologies:
    fail("training execution evidence bundle check: topology coverage drifted")

execution_classes = {
    execution_class
    for segment in fixture["segment_evidence"]
    for execution_class in segment["execution_classes"]
}
required_classes = {
    "dense_full_model_rank",
    "validated_contributor_window",
    "validator",
    "checkpoint_writer",
    "eval_worker",
}
if not required_classes.issubset(execution_classes):
    fail("training execution evidence bundle check: execution-class coverage drifted")

dispositions = {segment["segment_disposition"] for segment in fixture["segment_evidence"]}
if not {"completed_success", "degraded_success", "refused"}.issubset(dispositions):
    fail("training execution evidence bundle check: disposition coverage drifted")

summary = {
    "verdict": "verified",
    "segment_count": len(fixture["segment_evidence"]),
    "topology_kind_count": len(topology_kinds),
    "final_artifact_ref_count": len(fixture["final_artifact_refs"]),
}
print(json.dumps(summary, indent=2))
PY
