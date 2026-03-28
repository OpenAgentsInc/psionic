#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

run_id=""
bundle_dir=""
remote_host="archlinux"
remote_worktree_dir=""
remote_bundle_dir=""
topology_contract_rel="fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json"
workflow_plan_rel="fixtures/swarm/first_swarm_live_workflow_plan_v1.json"
git_ref=""
local_tailnet_ip=""
remote_tailnet_ip=""
coordinator_port="34100"
contributor_port="34101"

usage() {
  cat <<'EOF' >&2
Usage: scripts/run-first-swarm-tailnet-admitted-live.sh [options]

Options:
  --run-id <id>                 Stable run identifier. Default: tailrun-home-admitted-<utc>
  --bundle-dir <path>           Local output directory. Default: fixtures/swarm/runs/<run_id>
  --remote-host <host>          Remote Linux contributor SSH target. Default: archlinux
  --remote-worktree-dir <path>  Remote staged repo root. Default: $HOME/code/psionic-tailrun/<run_id>/repo
  --remote-bundle-dir <path>    Remote runtime output root. Default: $HOME/code/psionic-tailrun/<run_id>/linux
  --git-ref <ref>               Local git ref to archive for the remote run. Default: local HEAD
  --local-tailnet-ip <ip>       Explicit local Tailnet IPv4. Default: detect with tailscale ip -4
  --remote-tailnet-ip <ip>      Explicit remote Tailnet IPv4. Default: detect over SSH
  --coordinator-port <port>     Local coordinator port. Default: 34100
  --contributor-port <port>     Remote contributor port. Default: 34101
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
    --remote-worktree-dir)
      remote_worktree_dir="$2"
      shift 2
      ;;
    --remote-bundle-dir)
      remote_bundle_dir="$2"
      shift 2
      ;;
    --git-ref)
      git_ref="$2"
      shift 2
      ;;
    --local-tailnet-ip)
      local_tailnet_ip="$2"
      shift 2
      ;;
    --remote-tailnet-ip)
      remote_tailnet_ip="$2"
      shift 2
      ;;
    --coordinator-port)
      coordinator_port="$2"
      shift 2
      ;;
    --contributor-port)
      contributor_port="$2"
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
  run_id="tailrun-home-admitted-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${bundle_dir}" ]]; then
  bundle_dir="${repo_root}/fixtures/swarm/runs/${run_id}"
fi
mkdir -p "${bundle_dir}"
bundle_dir="$(cd "${bundle_dir}" && pwd)"

if [[ -z "${git_ref}" ]]; then
  git_ref="$(git -C "${repo_root}" rev-parse HEAD)"
fi

if [[ -z "${remote_worktree_dir}" ]]; then
  remote_worktree_dir="\$HOME/code/psionic-tailrun/${run_id}/repo"
fi

if [[ -z "${remote_bundle_dir}" ]]; then
  remote_bundle_dir="\$HOME/code/psionic-tailrun/${run_id}/linux"
fi

detect_local_tailnet_ip() {
  tailscale ip -4 2>/dev/null | awk 'NF { print; exit }'
}

detect_remote_tailnet_ip() {
  ssh "${remote_host}" "tailscale ip -4 2>/dev/null | awk 'NF { print; exit }'"
}

if [[ -z "${remote_tailnet_ip}" ]]; then
  remote_tailnet_ip="$(detect_remote_tailnet_ip)"
fi

if [[ -z "${remote_tailnet_ip}" ]]; then
  echo "error: failed to detect remote Tailnet IP for ${remote_host}" >&2
  exit 1
fi

if [[ -z "${local_tailnet_ip}" ]]; then
  local_tailnet_ip="$(detect_local_tailnet_ip)"
fi

if [[ -z "${local_tailnet_ip}" ]]; then
  echo "error: failed to detect local Tailnet IP" >&2
  exit 1
fi

mkdir -p "${bundle_dir}/logs"

coordinator_report_path="${bundle_dir}/coordinator_runtime_report.json"
contributor_report_path="${bundle_dir}/contributor_runtime_report.json"
operator_manifest_path="${bundle_dir}/operator_manifest.json"
bundle_path="${bundle_dir}/first_swarm_real_run_bundle.json"
summary_path="${bundle_dir}/tailrun_admitted_home_run_summary.json"
contributor_log_path="${bundle_dir}/logs/contributor.log"
coordinator_log_path="${bundle_dir}/logs/coordinator.log"

remote_contributor_report_path="${remote_bundle_dir}/contributor_runtime_report.json"
remote_target_dir='$HOME/code/psionic/target/first-swarm-tailnet'
remote_tmp_dir='$HOME/code/psionic-tailrun/tmp'

echo "Staging ${git_ref} to ${remote_host}:${remote_worktree_dir}"
git -C "${repo_root}" archive "${git_ref}" -- \
  Cargo.toml \
  Cargo.lock \
  crates \
  fixtures/attnres \
  fixtures/parameter_golf \
  fixtures/psion \
  fixtures/swarm \
  fixtures/tassadar/sources \
  fixtures/training | ssh "${remote_host}" "
  set -euo pipefail
  rm -rf ${remote_worktree_dir}
  mkdir -p ${remote_worktree_dir} ${remote_bundle_dir}
  tar -xf - -C ${remote_worktree_dir}
"

jq -n \
  --arg created_at "$(now_utc)" \
  --arg run_id "${run_id}" \
  --arg git_ref "${git_ref}" \
  --arg topology_rel "${topology_contract_rel}" \
  --arg workflow_rel "${workflow_plan_rel}" \
  --arg bundle_dir "${bundle_dir}" \
  --arg local_ip "${local_tailnet_ip}" \
  --arg coordinator_port "${coordinator_port}" \
  --arg remote_host "${remote_host}" \
  --arg remote_ip "${remote_tailnet_ip}" \
  --arg contributor_port "${contributor_port}" \
  '{
    schema_version: "swarm.first_trusted_tailnet_operator_manifest.v1",
    created_at_utc: $created_at,
    run_id: $run_id,
    git_ref: $git_ref,
    topology_contract_path: $topology_rel,
    workflow_plan_path: $workflow_rel,
    bundle_dir: $bundle_dir,
    coordinator: {
      host: "local",
      tailnet_ip: $local_ip,
      cluster_port: ($coordinator_port | tonumber),
      endpoint: "\($local_ip):\($coordinator_port)"
    },
    contributor: {
      host: $remote_host,
      tailnet_ip: $remote_ip,
      cluster_port: ($contributor_port | tonumber),
      endpoint: "\($remote_ip):\($contributor_port)"
    },
    claim_boundary: "This manifest records one admitted-device Tailnet operator run over the first-swarm trusted-home protocol. It freezes the endpoints, git revision, and retained bundle location, but does not itself claim contributor success, validator acceptance, aggregation, or publication."
  }' > "${operator_manifest_path}"

echo "Starting remote contributor on ${remote_host} (${remote_tailnet_ip}:${contributor_port})"
ssh "${remote_host}" \
  "bash -ic 'export CARGO_TARGET_DIR=${remote_target_dir}; export TMPDIR=${remote_tmp_dir}; mkdir -p \"${remote_target_dir}\" \"${remote_tmp_dir}\"; cd ${remote_worktree_dir} && cargo run -q -p psionic-train --bin first_swarm_trusted_lan_live_runtime -- --role contributor --run-id ${run_id} --topology-contract ${remote_worktree_dir}/${topology_contract_rel} --workflow-plan ${remote_worktree_dir}/${workflow_plan_rel} --local-endpoint ${remote_tailnet_ip}:${contributor_port} --peer-endpoint ${local_tailnet_ip}:${coordinator_port} --output ${remote_contributor_report_path}'" \
  >"${contributor_log_path}" 2>&1 &
contributor_ssh_pid=$!

cleanup() {
  if [[ -n "${contributor_ssh_pid:-}" ]] && kill -0 "${contributor_ssh_pid}" >/dev/null 2>&1; then
    kill "${contributor_ssh_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sleep 2

echo "Running local coordinator on ${local_tailnet_ip}:${coordinator_port}"
cargo run -q -p psionic-train --bin first_swarm_trusted_lan_live_runtime -- \
  --role coordinator \
  --run-id "${run_id}" \
  --topology-contract "${repo_root}/${topology_contract_rel}" \
  --workflow-plan "${repo_root}/${workflow_plan_rel}" \
  --local-endpoint "${local_tailnet_ip}:${coordinator_port}" \
  --peer-endpoint "${remote_tailnet_ip}:${contributor_port}" \
  --output "${coordinator_report_path}" \
  >"${coordinator_log_path}" 2>&1

wait "${contributor_ssh_pid}"
trap - EXIT

echo "Copying back remote contributor report"
scp "${remote_host}:${remote_contributor_report_path}" "${contributor_report_path}" >/dev/null

python3 - <<'PY' \
  "${operator_manifest_path}" "${coordinator_report_path}" "${contributor_report_path}" \
  "${bundle_path}" "${summary_path}" "${repo_root}/${topology_contract_rel}" \
  "${repo_root}/${workflow_plan_rel}"
import hashlib
import json
import os
import sys


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


def contribution_entry(report):
    local = report["local_contribution"]
    execution = local.get("execution_summary", {})
    started_at_ms = int(execution.get("started_at_ms", report["started_at_ms"]))
    completed_at_ms = int(execution.get("completed_at_ms", report["finished_at_ms"]))
    local_wallclock_ms = max(1, completed_at_ms - started_at_ms)
    sample_count = int(execution.get("sample_count", 0))
    step_count = int(execution.get("local_step_count", local.get("executed_steps", 0)))
    return {
        "node_id": report["node_id"],
        "runtime_role": report["runtime_role"],
        "role_id": local["role_id"],
        "execution_backend_label": report["execution_backend_label"],
        "endpoint": report["local_endpoint"],
        "observed_wallclock_ms": int(report["observed_wallclock_ms"]),
        "local_execution_wallclock_ms": local_wallclock_ms,
        "executed_steps": int(local["executed_steps"]),
        "batch_count": int(local["batch_count"]),
        "sample_count": sample_count,
        "payload_bytes": int(local["payload_bytes"]),
        "final_mean_loss": float(local["final_mean_loss"]),
        "contributor_receipt_digest": local["contributor_receipt"]["receipt_digest"],
        "estimated_steps_per_second": (step_count * 1000.0) / local_wallclock_ms,
        "estimated_samples_per_second": (sample_count * 1000.0) / local_wallclock_ms,
    }


manifest_path, coordinator_path, contributor_path, bundle_path, summary_path, topology_path, workflow_path = sys.argv[1:]
manifest = load(manifest_path)
coordinator = load(coordinator_path)
contributor = load(contributor_path)
topology = load(topology_path)
workflow = load(workflow_path)
retained_artifacts = coordinator.get("retained_artifacts")

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
if retained_artifacts is None:
    raise SystemExit("coordinator runtime report is missing retained_artifacts")

merged_report_path = retained_artifacts["merged_report_path"]
merged_report = load(merged_report_path)
if merged_report["schema_version"] != "swarm.first_trusted_lan_merged_artifact_report.v1":
    raise SystemExit("merged artifact report schema drifted")
if merged_report["merge_strategy"] != "exact_mean_delta_rank_stacking":
    raise SystemExit("merged artifact report merge_strategy drifted")

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
    publish_reason = "Publication is refused because the bounded admitted-device Tailnet run stopped at contributor, validator, replay, and aggregation truth without a promoted snapshot."

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
        "local_contributor_adapter_path": os.path.abspath(retained_artifacts["local_contributor_adapter_path"]),
        "remote_contributor_adapter_path": os.path.abspath(retained_artifacts["remote_contributor_adapter_path"]),
        "merged_adapter_path": os.path.abspath(retained_artifacts["merged_adapter_path"]),
        "merged_portable_bundle_path": os.path.abspath(retained_artifacts["merged_portable_bundle_path"]),
        "merged_report_path": os.path.abspath(merged_report_path),
    },
    "merged_artifact": {
        "merge_strategy": merged_report["merge_strategy"],
        "merged_lora_rank": int(merged_report["merged_lora_rank"]),
        "merged_lora_alpha": float(merged_report["merged_lora_alpha"]),
        "merged_portable_bundle_state_dict_digest": merged_report["merged_portable_bundle_state_dict_digest"],
        "merged_portable_bundle_artifact_digest": merged_report["merged_portable_bundle_artifact_digest"],
        "canonical_profile_mean_loss": float(merged_report["canonical_profile_mean_loss"]),
        "canonical_profile_bits_per_token": float(merged_report["canonical_profile_bits_per_token"]),
        "deterministic_probe_top_token_id": int(merged_report["deterministic_probe_top_token_id"]),
    },
    "claim_boundary": "This bundle proves one real admitted-device Tailnet mixed-hardware open-adapter run across a local M5 MLX coordinator and a Linux RTX 4080 contributor, with explicit contributor receipts, submission receipts, validator summary, replay receipts, aggregation outcome, and one inferable exact mean-delta merged adapter plus portable bundle. It does not claim M2 participation, exact HOMEGOLF or Parameter Golf score parity, full-model mixed-backend dense training, or automatic published-model promotion.",
}
encoded = json.dumps(bundle, indent=2)
bundle["bundle_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
with open(bundle_path, "w", encoding="utf-8") as handle:
    json.dump(bundle, handle, indent=2)
    handle.write("\n")

contributions = [contribution_entry(coordinator), contribution_entry(contributor)]
total_samples = sum(entry["sample_count"] for entry in contributions) or 1
for entry in contributions:
    entry["contribution_share"] = entry["sample_count"] / total_samples

tailrun_summary = {
    "schema_version": "swarm.tailrun_admitted_home_run_summary.v1",
    "run_id": coordinator["run_id"],
    "run_family_id": coordinator["run_family_id"],
    "result_classification": bundle["result_classification"],
    "admitted_device_set": [
        "local_m5_mlx",
        "archlinux_rtx4080_cuda",
    ],
    "accepted_contributions": int(summary["accepted_contributions"]),
    "replay_checked_contributions": int(summary["replay_checked_contributions"]),
    "submission_receipt_count": submission_count,
    "merge_disposition": merge_disposition,
    "publish_disposition": publish_disposition,
    "promotion_disposition": promotion["promotion_disposition"],
    "per_device_contributions": contributions,
    "artifacts": {
        "operator_manifest_path": os.path.abspath(manifest_path),
        "bundle_path": os.path.abspath(bundle_path),
        "coordinator_report_path": os.path.abspath(coordinator_path),
        "contributor_report_path": os.path.abspath(contributor_path),
        "merged_portable_bundle_path": os.path.abspath(retained_artifacts["merged_portable_bundle_path"]),
        "merged_report_path": os.path.abspath(merged_report_path),
    },
    "merged_artifact": {
        "merge_strategy": merged_report["merge_strategy"],
        "merged_lora_rank": int(merged_report["merged_lora_rank"]),
        "canonical_profile_mean_loss": float(merged_report["canonical_profile_mean_loss"]),
        "deterministic_probe_top_token_id": int(merged_report["deterministic_probe_top_token_id"]),
    },
    "claim_boundary": "This summary records per-device contribution accounting for one admitted-device home-Tailnet run plus one retained inferable exact mean-delta merged adapter and portable bundle. It does not claim M2 participation, open internet swarm membership, full dense training parity, or public Parameter Golf score equivalence.",
}
encoded = json.dumps(tailrun_summary, indent=2)
tailrun_summary["summary_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(tailrun_summary, handle, indent=2)
    handle.write("\n")
PY

echo "Wrote live bundle ${bundle_path}"
echo "Wrote Tailnet summary ${summary_path}"
echo "Use: scripts/check-first-swarm-trusted-lan-real-run.sh --bundle ${bundle_path}"
