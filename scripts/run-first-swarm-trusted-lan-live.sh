#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

run_id=""
bundle_dir=""
remote_host="archlinux"
remote_repo_dir="~/code/psionic"
remote_worktree_dir="~/code/psionic-swarm-first-live"
topology_contract_rel="fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json"
workflow_plan_rel="fixtures/swarm/first_swarm_live_workflow_plan_v1.json"
git_ref=""
local_lan_ip=""
remote_lan_ip=""
coordinator_port="34100"
contributor_port="34101"

usage() {
  cat <<'EOF' >&2
Usage: scripts/run-first-swarm-trusted-lan-live.sh [options]

Options:
  --run-id <id>                 Stable run identifier. Default: first-swarm-live-<utc>
  --bundle-dir <path>           Local output directory. Default: fixtures/swarm/runs/<run_id>
  --remote-host <host>          Remote Linux contributor SSH target. Default: archlinux
  --remote-repo-dir <path>      Remote psionic repo used for git fetch/worktree. Default: ~/code/psionic
  --remote-worktree-dir <path>  Clean remote worktree used for the live run. Default: ~/code/psionic-swarm-first-live
  --git-ref <ref>               Git ref the remote worktree should check out. Default: local HEAD
  --local-lan-ip <ip>           Explicit local LAN IP for the coordinator. Default: auto-detect
  --remote-lan-ip <ip>          Explicit remote LAN IP for the contributor. Default: auto-detect over SSH
  --topology-contract <path>    Repo-relative topology contract path. Default: fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json
  --workflow-plan <path>        Repo-relative workflow plan path. Default: fixtures/swarm/first_swarm_live_workflow_plan_v1.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --bundle-dir)
      bundle_dir="$2"
      shift 2
      ;;
    --remote-host)
      remote_host="$2"
      shift 2
      ;;
    --remote-repo-dir)
      remote_repo_dir="$2"
      shift 2
      ;;
    --remote-worktree-dir)
      remote_worktree_dir="$2"
      shift 2
      ;;
    --git-ref)
      git_ref="$2"
      shift 2
      ;;
    --local-lan-ip)
      local_lan_ip="$2"
      shift 2
      ;;
    --remote-lan-ip)
      remote_lan_ip="$2"
      shift 2
      ;;
    --topology-contract)
      topology_contract_rel="$2"
      shift 2
      ;;
    --workflow-plan)
      workflow_plan_rel="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

now_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if [[ -z "${run_id}" ]]; then
  run_id="first-swarm-live-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${bundle_dir}" ]]; then
  bundle_dir="${repo_root}/fixtures/swarm/runs/${run_id}"
fi

if [[ -z "${git_ref}" ]]; then
  git_ref="$(git -C "${repo_root}" rev-parse HEAD)"
fi

bundle_dir="$(python3 - <<'PY' "${bundle_dir}"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"

detect_local_ip() {
  python3 - <<'PY' "$1"
import socket, sys
target = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect((target, 1))
print(s.getsockname()[0])
s.close()
PY
}

if [[ -z "${remote_lan_ip}" ]]; then
  remote_lan_ip="$(
    ssh "${remote_host}" "ip route get 1.1.1.1 | awk '/src/ {for (i = 1; i <= NF; i++) if (\$i == \"src\") { print \$(i+1); exit }}'"
  )"
fi

if [[ -z "${remote_lan_ip}" ]]; then
  echo "error: failed to detect remote LAN IP for ${remote_host}" >&2
  exit 1
fi

if [[ -z "${local_lan_ip}" ]]; then
  local_lan_ip="$(detect_local_ip "${remote_lan_ip}")"
fi

if [[ -z "${local_lan_ip}" ]]; then
  echo "error: failed to detect local LAN IP" >&2
  exit 1
fi

mkdir -p "${bundle_dir}/logs"

coordinator_report_path="${bundle_dir}/coordinator_runtime_report.json"
contributor_report_path="${bundle_dir}/contributor_runtime_report.json"
operator_manifest_path="${bundle_dir}/operator_manifest.json"
bundle_path="${bundle_dir}/first_swarm_real_run_bundle.json"
contributor_log_path="${bundle_dir}/logs/contributor.log"
coordinator_log_path="${bundle_dir}/logs/coordinator.log"

remote_bundle_dir="~/swarm-runs/${run_id}/linux"
remote_contributor_report_path="${remote_bundle_dir}/contributor_runtime_report.json"

remote_repo_dir_escaped="${remote_repo_dir}"
remote_worktree_dir_escaped="${remote_worktree_dir}"
remote_bundle_dir_escaped="${remote_bundle_dir}"
remote_topology_path="${remote_worktree_dir_escaped}/${topology_contract_rel}"
remote_workflow_path="${remote_worktree_dir_escaped}/${workflow_plan_rel}"

echo "Preparing remote worktree ${remote_worktree_dir} at ${git_ref}"
ssh "${remote_host}" "
  set -euo pipefail
  cd ${remote_repo_dir_escaped}
  git fetch origin
  git worktree remove -f ${remote_worktree_dir_escaped} >/dev/null 2>&1 || true
  rm -rf ${remote_worktree_dir_escaped}
  git worktree add -f ${remote_worktree_dir_escaped} ${git_ref}
  mkdir -p ${remote_bundle_dir_escaped}
"

python3 - <<'PY' \
  "${operator_manifest_path}" "${run_id}" "${git_ref}" "${topology_contract_rel}" "${workflow_plan_rel}" \
  "${local_lan_ip}" "${coordinator_port}" "${remote_lan_ip}" "${contributor_port}" "${remote_host}" "${bundle_dir}"
import json, os, sys

path, run_id, git_ref, topology_rel, workflow_rel, local_ip, local_port, remote_ip, remote_port, remote_host, bundle_dir = sys.argv[1:]
doc = {
    "schema_version": "swarm.first_trusted_lan_operator_manifest.v1",
    "created_at_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "run_id": run_id,
    "git_ref": git_ref,
    "topology_contract_path": topology_rel,
    "workflow_plan_path": workflow_rel,
    "bundle_dir": bundle_dir,
    "coordinator": {
        "host": "local",
        "lan_ip": local_ip,
        "cluster_port": int(local_port),
        "endpoint": f"{local_ip}:{local_port}",
    },
    "contributor": {
        "host": remote_host,
        "lan_ip": remote_ip,
        "cluster_port": int(remote_port),
        "endpoint": f"{remote_ip}:{remote_port}",
    },
    "claim_boundary": "This manifest records the exact operator-selected endpoints and git revision for one real first swarm trusted-LAN attempt. It does not itself claim contributor success, validator acceptance, aggregation, or publication.",
}
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
PY

echo "Starting remote contributor on ${remote_host} (${remote_lan_ip}:${contributor_port})"
ssh "${remote_host}" \
  "bash -ic 'cd ${remote_worktree_dir_escaped} && cargo run -q -p psionic-train --bin first_swarm_trusted_lan_live_runtime -- --role contributor --run-id ${run_id} --topology-contract ${remote_topology_path} --workflow-plan ${remote_workflow_path} --local-endpoint ${remote_lan_ip}:${contributor_port} --peer-endpoint ${local_lan_ip}:${coordinator_port} --output ${remote_contributor_report_path}'" \
  >"${contributor_log_path}" 2>&1 &
contributor_ssh_pid=$!

cleanup() {
  if [[ -n "${contributor_ssh_pid:-}" ]] && kill -0 "${contributor_ssh_pid}" >/dev/null 2>&1; then
    kill "${contributor_ssh_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sleep 2

echo "Running local coordinator on ${local_lan_ip}:${coordinator_port}"
cargo run -q -p psionic-train --bin first_swarm_trusted_lan_live_runtime -- \
  --role coordinator \
  --run-id "${run_id}" \
  --topology-contract "${repo_root}/${topology_contract_rel}" \
  --workflow-plan "${repo_root}/${workflow_plan_rel}" \
  --local-endpoint "${local_lan_ip}:${coordinator_port}" \
  --peer-endpoint "${remote_lan_ip}:${contributor_port}" \
  --output "${coordinator_report_path}" \
  >"${coordinator_log_path}" 2>&1

wait "${contributor_ssh_pid}"
trap - EXIT

echo "Copying back remote contributor report"
scp "${remote_host}:${remote_contributor_report_path}" "${contributor_report_path}" >/dev/null

python3 - <<'PY' \
  "${operator_manifest_path}" "${coordinator_report_path}" "${contributor_report_path}" \
  "${bundle_path}" "${repo_root}/${topology_contract_rel}" "${repo_root}/${workflow_plan_rel}"
import hashlib, json, os, sys

manifest_path, coordinator_path, contributor_path, bundle_path, topology_path, workflow_path = sys.argv[1:]

def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

manifest = load(manifest_path)
coordinator = load(coordinator_path)
contributor = load(contributor_path)
topology = load(topology_path)
workflow = load(workflow_path)

if coordinator["runtime_role"] != "coordinator":
    raise SystemExit("coordinator runtime report does not carry runtime_role=coordinator")
if contributor["runtime_role"] != "contributor":
    raise SystemExit("contributor runtime report does not carry runtime_role=contributor")
if coordinator["run_id"] != contributor["run_id"]:
    raise SystemExit("runtime reports disagree on run_id")
if coordinator["execution_backend_label"] != "open_adapter_backend.mlx.metal.gpt_oss_lm_head":
    raise SystemExit("coordinator report does not preserve the MLX backend label")
if contributor["execution_backend_label"] != "open_adapter_backend.cuda.gpt_oss_lm_head":
    raise SystemExit("contributor report does not preserve the CUDA backend label")
if coordinator.get("validator_summary") is None:
    raise SystemExit("coordinator runtime report is missing validator_summary")
if coordinator.get("promotion_receipt") is None:
    raise SystemExit("coordinator runtime report is missing promotion_receipt")
if coordinator.get("aggregation_compatibility") is None:
    raise SystemExit("coordinator runtime report is missing aggregation_compatibility")

summary = coordinator["validator_summary"]
promotion = coordinator["promotion_receipt"]
accepted = int(summary["accepted_contributions"])
replay_checked = int(summary["replay_checked_contributions"])
submission_count = len(coordinator.get("submission_receipts", []))
merge_disposition = "merged" if accepted == 2 and submission_count == 2 else "no_merge"
publish_disposition = "refused"
if promotion.get("promotion_disposition") == "promoted":
    publish_reason = "The runtime promoted an aggregated local snapshot candidate, but this operator run did not execute the later publish surface."
else:
    publish_reason = "Publication is refused because the bounded live run stopped at contributor, validator, replay, and aggregation truth without a promoted snapshot."

bundle = {
    "schema_version": "swarm.first_trusted_lan_real_run_bundle.v1",
    "run_id": coordinator["run_id"],
    "run_family_id": coordinator["run_family_id"],
    "result_classification": "bounded_success" if accepted == 2 and replay_checked == 2 and submission_count == 2 else "partial_success",
    "operator_manifest_sha256": sha256_file(manifest_path),
    "topology_contract_digest": topology["contract_digest"],
    "workflow_plan_digest": workflow["plan_digest"],
    "coordinator_report_sha256": sha256_file(coordinator_path),
    "contributor_report_sha256": sha256_file(contributor_path),
    "coordinator_endpoint": coordinator["local_endpoint"],
    "contributor_endpoint": contributor["local_endpoint"],
    "coordinator_backend_label": coordinator["execution_backend_label"],
    "contributor_backend_label": contributor["execution_backend_label"],
    "coordinator_contributor_receipt_digest": coordinator["local_contribution"]["contributor_receipt"]["receipt_digest"],
    "contributor_contributor_receipt_digest": contributor["local_contribution"]["contributor_receipt"]["receipt_digest"],
    "aggregation_compatibility_digest": hashlib.sha256(json.dumps(coordinator["aggregation_compatibility"], sort_keys=True).encode("utf-8")).hexdigest(),
    "validator_summary_digest": summary["summary_digest"],
    "promotion_receipt_digest": promotion["receipt_digest"],
    "promotion_disposition": promotion["promotion_disposition"],
    "promotion_hold_reason_codes": promotion.get("hold_reason_codes", []),
    "total_contributions": summary["total_contributions"],
    "accepted_contributions": summary["accepted_contributions"],
    "replay_checked_contributions": summary["replay_checked_contributions"],
    "submission_receipt_count": submission_count,
    "replay_receipt_digests": coordinator.get("replay_receipt_digests", []),
    "merge_disposition": merge_disposition,
    "publish_disposition": publish_disposition,
    "publish_reason": publish_reason,
    "artifacts": {
        "operator_manifest_path": os.path.abspath(manifest_path),
        "topology_contract_path": os.path.abspath(topology_path),
        "workflow_plan_path": os.path.abspath(workflow_path),
        "coordinator_report_path": os.path.abspath(coordinator_path),
        "contributor_report_path": os.path.abspath(contributor_path),
    },
    "claim_boundary": "This bundle proves one real trusted-LAN mixed-hardware open-adapter run across a Mac MLX coordinator and a Linux RTX 4080 contributor, with explicit contributor receipts, submission receipts, validator summary, replay receipts, and aggregation outcome. It does not claim full-model mixed-backend dense training or automatic published-model promotion.",
}
encoded = json.dumps(bundle, indent=2)
bundle["bundle_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
with open(bundle_path, "w", encoding="utf-8") as handle:
    json.dump(bundle, handle, indent=2)
    handle.write("\n")
PY

echo "Wrote live bundle ${bundle_path}"
echo "Use: scripts/check-first-swarm-trusted-lan-real-run.sh --bundle ${bundle_path}"
