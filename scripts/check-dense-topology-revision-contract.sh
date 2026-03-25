#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/dense_topology_revision_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/dense_topology_revision_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/dense_topology_revision_contract_v1.json"
cargo run -q -p psionic-train --bin dense_topology_revision_contract -- "${generated_path}" >/dev/null

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
    fail("dense topology-revision contract check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.dense_topology_revision_contract.v1":
    fail("dense topology-revision contract check: schema version drifted")

if len(fixture["revisions"]) != 4:
    fail("dense topology-revision contract check: revision count drifted")

supported = [revision for revision in fixture["revisions"] if revision["disposition"] == "supported"]
refused = [revision for revision in fixture["revisions"] if revision["disposition"] == "refused"]
if len(supported) != 3 or len(refused) != 1:
    fail("dense topology-revision contract check: supported/refused split drifted")

grow = next(revision for revision in fixture["revisions"] if revision["action_kind"] == "grow_world")
shrink = next(revision for revision in fixture["revisions"] if revision["action_kind"] == "shrink_world")
replace = next(revision for revision in fixture["revisions"] if revision["action_kind"] == "replace_rank")

if grow["data_ordering"]["policy_kind"] != "checkpoint_barrier_reseed":
    fail("dense topology-revision contract check: grow-world lost checkpoint-barrier reseed policy")
if shrink["data_ordering"]["policy_kind"] != "checkpoint_barrier_reseed":
    fail("dense topology-revision contract check: shrink-world lost checkpoint-barrier reseed policy")
if replace["data_ordering"]["policy_kind"] != "replay_continuation":
    fail("dense topology-revision contract check: replace-rank lost replay-continuation policy")

summary = {
    "verdict": "verified",
    "revision_count": len(fixture["revisions"]),
    "supported_count": len(supported),
    "refused_count": len(refused),
    "whole_program_run_graph_digest": fixture["whole_program_run_graph_digest"],
}
print(json.dumps(summary, indent=2))
PY
