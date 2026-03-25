#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
bundle_dir=""

usage() {
  cat <<'EOF' >&2
Usage: check-first-swarm-trusted-lan.sh [--bundle-dir <path>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      bundle_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cleanup_bundle=0
if [[ -z "${bundle_dir}" ]]; then
  bundle_dir="$(mktemp -d "${TMPDIR:-/tmp}/first_swarm_trusted_lan_bundle.XXXXXX")"
  cleanup_bundle=1
fi

cleanup() {
  if [[ "${cleanup_bundle}" -eq 1 ]]; then
    rm -rf -- "${bundle_dir}"
  fi
}
trap cleanup EXIT

manifest_output="${bundle_dir}/launch.out"
bash "${repo_root}/scripts/first-swarm-launch-trusted-lan.sh" \
  --run-id "first-swarm-trusted-lan-check" \
  --bundle-dir "${bundle_dir}" \
  --manifest-only > "${manifest_output}"

python3 - "${bundle_dir}" "${manifest_output}" <<'PY'
import json
import sys
from pathlib import Path

bundle_dir = Path(sys.argv[1])
manifest_output = Path(sys.argv[2])

manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
topology = json.loads((bundle_dir / "first_swarm_trusted_lan_topology_contract_v1.json").read_text(encoding="utf-8"))
failure_drills = json.loads((bundle_dir / "reports" / "first_swarm_trusted_lan_failure_drills_v1.json").read_text(encoding="utf-8"))
workflow_plan = json.loads((bundle_dir / "first_swarm_live_workflow_plan_v1.json").read_text(encoding="utf-8"))
receipt = json.loads((bundle_dir / "first_swarm_trusted_lan_launch_receipt.json").read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if manifest["schema_version"] != "swarm.first_trusted_lan_launch_manifest.v1":
    fail("first swarm trusted-lan check: manifest schema version drifted")
if manifest["launcher"]["execution_posture"] != "local_bundle_materialization_only":
    fail("first swarm trusted-lan check: launcher execution posture drifted")
if manifest["contract_digests"]["topology_contract_digest"] != topology["contract_digest"]:
    fail("first swarm trusted-lan check: topology digest mismatch between manifest and topology contract")
if manifest["contract_digests"]["failure_drills_digest"] != failure_drills["bundle_digest"]:
    fail("first swarm trusted-lan check: failure-drill digest mismatch between manifest and failure-drill bundle")
if manifest["contract_digests"]["workflow_plan_digest"] != workflow_plan["plan_digest"]:
    fail("first swarm trusted-lan check: workflow-plan digest mismatch between manifest and workflow plan")
if topology["schema_version"] != "swarm.first_trusted_lan_topology_contract.v1":
    fail("first swarm trusted-lan check: topology schema version drifted")
if topology["cluster_namespace"] != "cluster.swarm.local.trusted_lan":
    fail("first swarm trusted-lan check: topology lost the trusted-LAN cluster namespace")
if topology["admission_token_env_var"] != "PSIONIC_SWARM_ADMISSION_TOKEN":
    fail("first swarm trusted-lan check: topology lost the admission token env var")
if topology["heartbeat_policy"]["heartbeat_interval_ms"] != 1000:
    fail("first swarm trusted-lan check: heartbeat interval drifted")
if topology["heartbeat_policy"]["stale_after_ms"] != 5000:
    fail("first swarm trusted-lan check: stale-worker threshold drifted")
if topology["heartbeat_policy"]["max_worker_skew_ms"] != 15000:
    fail("first swarm trusted-lan check: worker-skew threshold drifted")
if len(topology["nodes"]) != 2:
    fail("first swarm trusted-lan check: topology must retain exactly two nodes")
if topology["coordinator_node_id"] != "swarm-mac-a":
    fail("first swarm trusted-lan check: coordinator node id drifted")
node_ids = {node["node_id"] for node in topology["nodes"]}
if node_ids != {"swarm-mac-a", "swarm-linux-4080-a"}:
    fail("first swarm trusted-lan check: topology node set drifted")
if len(topology["launch_sequence"]) < 5:
    fail("first swarm trusted-lan check: launch sequence lost one or more required phases")
if "check-first-swarm-trusted-lan.sh" not in topology["check_script_path"]:
    fail("first swarm trusted-lan check: topology checker path drifted")
if failure_drills["schema_version"] != "swarm.first_trusted_lan_failure_drills.v1":
    fail("first swarm trusted-lan check: failure-drill schema version drifted")
if failure_drills["topology_contract_digest"] != topology["contract_digest"]:
    fail("first swarm trusted-lan check: failure-drill bundle lost topology digest linkage")
required_drill_kinds = {"stale_worker", "upload_disagreement", "contributor_loss", "uneven_worker_speed"}
drill_kinds = {drill["drill_kind"] for drill in failure_drills["drills"]}
if drill_kinds != required_drill_kinds:
    fail("first swarm trusted-lan check: failure-drill bundle does not cover the required drill kinds")
stale_worker = next(drill for drill in failure_drills["drills"] if drill["drill_kind"] == "stale_worker")
if stale_worker["validator_disposition"] != "replay_required":
    fail("first swarm trusted-lan check: stale-worker drill must stay replay_required")
upload_disagreement = next(drill for drill in failure_drills["drills"] if drill["drill_kind"] == "upload_disagreement")
if upload_disagreement["validator_disposition"] != "rejected":
    fail("first swarm trusted-lan check: upload disagreement drill must stay rejected")
if upload_disagreement.get("expected_upload_manifest_digest") == upload_disagreement.get("observed_upload_manifest_digest"):
    fail("first swarm trusted-lan check: upload disagreement drill must keep mismatched digests explicit")
if receipt["launch_status"] != "bundle_materialized":
    fail("first swarm trusted-lan check: launch receipt status drifted")
phase_results = receipt["phase_results"]
if len(phase_results) != 5:
    fail("first swarm trusted-lan check: expected five local bundle-materialization phases")
if any(result["status"] != "completed" for result in phase_results):
    fail("first swarm trusted-lan check: all local phases must complete successfully")
remote_commands = manifest["remote_commands"]
if "check-swarm-mac-mlx-bringup.sh" not in remote_commands["mac_bringup_command"]:
    fail("first swarm trusted-lan check: manifest lost the Mac bring-up command")
if "check-swarm-linux-4080-bringup.sh" not in remote_commands["linux_bringup_command"]:
    fail("first swarm trusted-lan check: manifest lost the Linux bring-up command")
if "first_swarm_live_workflow_plan" not in remote_commands["workflow_plan_command"]:
    fail("first swarm trusted-lan check: manifest lost the workflow-plan command")
if "first_swarm_trusted_lan_failure_drills" not in remote_commands["failure_drills_command"]:
    fail("first swarm trusted-lan check: manifest lost the failure-drills command")

summary = {
    "verdict": "verified",
    "run_family_id": manifest["run_family_id"],
    "topology_contract_digest": topology["contract_digest"],
    "failure_drills_digest": failure_drills["bundle_digest"],
    "workflow_plan_digest": workflow_plan["plan_digest"],
    "phase_count": len(phase_results),
    "node_ids": sorted(node_ids),
}
print(json.dumps(summary, indent=2))
PY
