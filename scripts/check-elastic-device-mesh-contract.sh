#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/elastic_device_mesh_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/elastic_device_mesh_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/elastic_device_mesh_contract_v1.json"
cargo run -q -p psionic-train --bin elastic_device_mesh_contract -- "${generated_path}" >/dev/null

python3 - "${contract_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

committed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generated = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if committed != generated:
    fail("elastic device mesh contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.elastic_device_mesh_contract.v1":
    fail("elastic device mesh contract check: schema_version drifted")
if committed["contract_id"] != "psionic.elastic_device_mesh_contract.v1":
    fail("elastic device mesh contract check: contract_id drifted")
if committed["current_epoch_id"] != "network_epoch_00000123":
    fail("elastic device mesh contract check: current_epoch_id drifted")
if len(committed["role_lease_policies"]) != 5:
    fail("elastic device mesh contract check: expected exactly five role lease policies")
if len(committed["member_leases"]) != 8:
    fail("elastic device mesh contract check: expected exactly eight member leases")
if len(committed["heartbeat_samples"]) != 8:
    fail("elastic device mesh contract check: expected exactly eight heartbeat samples")
if len(committed["deathrattles"]) != 1:
    fail("elastic device mesh contract check: expected exactly one deathrattle")
if len(committed["revision_receipts"]) != 5:
    fail("elastic device mesh contract check: expected exactly five revision receipts")

deathrattle = committed["deathrattles"][0]
if deathrattle["replacement_registry_record_id"] != "local_mlx_mac_workstation.registry":
    fail("elastic device mesh contract check: deathrattle replacement drifted")

refusal = next(
    revision for revision in committed["revision_receipts"]
    if revision["revision_kind"] == "refuse_dense_world_change"
)
if refusal["outcome"] != "refused":
    fail("elastic device mesh contract check: dense world refusal outcome drifted")
if refusal["referenced_topology_revision_id"] != "dense_topology.remove_rank_without_replacement.live_refused":
    fail("elastic device mesh contract check: topology refusal binding drifted")
if refusal["referenced_recovery_scenario_id"] != "dense_rank.provider_loss.rank3.shrink_world_refused":
    fail("elastic device mesh contract check: recovery refusal binding drifted")

if committed["authority_paths"]["check_script_path"] != "scripts/check-elastic-device-mesh-contract.sh":
    fail("elastic device mesh contract check: check_script_path drifted")
if committed["authority_paths"]["reference_doc_path"] != "docs/ELASTIC_DEVICE_MESH_REFERENCE.md":
    fail("elastic device mesh contract check: reference_doc_path drifted")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "lease_ids": [lease["lease_id"] for lease in committed["member_leases"]],
    "revision_ids": [revision["revision_id"] for revision in committed["revision_receipts"]],
}
print(json.dumps(summary, indent=2))
PY
