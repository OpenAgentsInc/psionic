#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/contributor_program_lineage_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/contributor_program_lineage.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/contributor_program_lineage_v1.json"
cargo run -q -p psionic-train --bin contributor_program_lineage -- "${generated_path}" >/dev/null

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
    fail("contributor program-lineage check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.contributor_program_lineage.v1":
    fail("contributor program-lineage check: schema version drifted")

if fixture["dataset_family_id"] != "psion.curated_pretrain.dataset_family.v1":
    fail("contributor program-lineage check: dataset family drifted")
if fixture["checkpoint_family"] != "psion.cross_provider.pretrain.v1":
    fail("contributor program-lineage check: checkpoint family drifted")

if len(fixture["contributor_window_bindings"]) != 3:
    fail("contributor program-lineage check: contributor window count drifted")
if len(fixture["promotion_contracts"]) != 3:
    fail("contributor program-lineage check: promotion contract count drifted")

for binding in fixture["contributor_window_bindings"]:
    if binding["input_policy_revision_id"] != fixture["input_policy_revision"]["revision_id"]:
        fail(f"contributor program-lineage check: binding {binding['window_id']} lost shared policy revision")

for contract in fixture["promotion_contracts"]:
    if contract["input_policy_revision"]["revision_id"] != fixture["input_policy_revision"]["revision_id"]:
        fail(f"contributor program-lineage check: promotion contract {contract['promotion_contract_id']} lost shared policy revision")

summary = {
    "verdict": "verified",
    "contributor_window_count": len(fixture["contributor_window_bindings"]),
    "promotion_contract_count": len(fixture["promotion_contracts"]),
    "input_policy_revision_id": fixture["input_policy_revision"]["revision_id"],
}
print(json.dumps(summary, indent=2))
PY
