#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/sharded_distributed_checkpoint_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/sharded_distributed_checkpoint_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/sharded_distributed_checkpoint_contract_v1.json"
cargo run -q -p psionic-train --bin sharded_distributed_checkpoint_contract -- "${generated_path}" >/dev/null

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
    fail("distributed checkpoint contract check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.sharded_distributed_checkpoint_contract.v1":
    fail("distributed checkpoint contract check: schema version drifted")
if fixture["checkpoint_manifest"]["checkpoint_family"] != "psion.cross_provider.pretrain.v1":
    fail("distributed checkpoint contract check: checkpoint family drifted")
if fixture["restore_plan"]["manifest_authority_source_id"] != "google_l4_validator_node":
    fail("distributed checkpoint contract check: manifest authority drifted")

placements = fixture["shard_placements"]
if len(placements) != 16:
    fail("distributed checkpoint contract check: shard placement count drifted")

parameter_count = sum(1 for placement in placements if placement["shard_role"] == "parameter_state")
optimizer_count = sum(1 for placement in placements if placement["shard_role"] == "optimizer_state")
if parameter_count != 8 or optimizer_count != 8:
    fail("distributed checkpoint contract check: parameter/optimizer shard counts drifted")

receipts = fixture["shard_upload_receipts"]
durable_count = sum(1 for receipt in receipts if receipt["upload_disposition"] == "durable")
refused_count = sum(1 for receipt in receipts if receipt["upload_disposition"] == "partial_upload_refused")
if durable_count != 16 or refused_count < 1:
    fail("distributed checkpoint contract check: upload receipt durability posture drifted")

assignments = fixture["restore_plan"]["assignments"]
if len(assignments) != 8:
    fail("distributed checkpoint contract check: restore assignment count drifted")
if any(assignment["requested_execution_class"] != "dense_full_model_rank" for assignment in assignments):
    fail("distributed checkpoint contract check: restore execution class drifted")

summary = {
    "verdict": "verified",
    "shard_count": len(placements),
    "durable_upload_receipt_count": durable_count,
    "restore_assignment_count": len(assignments),
}
print(json.dumps(summary, indent=2))
PY
