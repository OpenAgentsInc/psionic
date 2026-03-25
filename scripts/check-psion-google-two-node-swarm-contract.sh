#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/psion/google/psion_google_two_node_swarm_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/psion_google_two_node_swarm_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/psion_google_two_node_swarm_contract_v1.json"
cargo run -q -p psionic-train --bin psion_google_two_node_swarm_contract -- "${generated_path}" >/dev/null

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
    fail("google two-node swarm contract check: committed fixture drifted from the generator output")
if committed["schema_version"] != "psion.google_two_node_swarm_contract.v1":
    fail("google two-node swarm contract check: schema_version drifted")
if committed["project_id"] != "openagentsgemini":
    fail("google two-node swarm contract check: project_id drifted")
if committed["cluster_admission_posture"] != "authenticated_configured_peers.operator_manifest":
    fail("google two-node swarm contract check: cluster admission posture drifted")
if committed["discovery_posture"] != "configured_peer_only":
    fail("google two-node swarm contract check: discovery posture drifted")
if committed["external_ip_permitted"] is not False:
    fail("google two-node swarm contract check: lane must keep external IPs disabled")
if len(committed["nodes"]) != 2:
    fail("google two-node swarm contract check: lane must keep exactly two nodes")
zones = {node["preferred_zone"] for node in committed["nodes"]}
if len(zones) != 2:
    fail("google two-node swarm contract check: nodes must stay in different zones")
subnetworks = {node["subnetwork"] for node in committed["nodes"]}
if len(subnetworks) != 2:
    fail("google two-node swarm contract check: nodes must stay on distinct dedicated subnetworks")
if {node["role_kind"] for node in committed["nodes"]} != {
    "coordinator_validator_aggregator_contributor",
    "contributor",
}:
    fail("google two-node swarm contract check: node role kinds drifted")
if any(node["backend_label"] != "open_adapter_backend.cuda.gpt_oss_lm_head" for node in committed["nodes"]):
    fail("google two-node swarm contract check: backend label drifted away from the CUDA open-adapter lane")
expected_profiles = {
    "clean_baseline",
    "mild_wan",
    "asymmetric_degraded",
    "temporary_partition",
}
if set(committed["admitted_impairment_profile_ids"]) != expected_profiles:
    fail("google two-node swarm contract check: impairment profile set drifted")
expected_result_classes = {
    "configured_peer_launch_failure",
    "cluster_membership_failure",
    "network_impairment_gate_failure",
    "contributor_execution_failure",
    "validator_refusal",
    "aggregation_failure",
    "bounded_success",
}
if set(committed["result_classifications"]) != expected_result_classes:
    fail("google two-node swarm contract check: result classification set drifted")
if committed["artifact_authority"]["launch_profiles_path"] != "fixtures/psion/google/psion_google_two_node_swarm_launch_profiles_v1.json":
    fail("google two-node swarm contract check: reserved launch_profiles_path drifted")
if committed["artifact_authority"]["runbook_path"] != "docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md":
    fail("google two-node swarm contract check: reserved runbook path drifted")
if committed["bucket_authority"]["cluster_manifest_object"] != "launch/psion_google_two_node_swarm_cluster_manifest.json":
    fail("google two-node swarm contract check: cluster manifest object drifted")
if len(committed["baseline_single_node_artifacts"]) != 4:
    fail("google two-node swarm contract check: baseline single-node artifact linkage drifted")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "zone_pairs": [pair["pair_id"] for pair in committed["admitted_zone_pairs"]],
    "node_ids": [node["node_id"] for node in committed["nodes"]],
}
print(json.dumps(summary, indent=2))
PY
