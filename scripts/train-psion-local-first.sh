#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

mode="auto"
workload="reference_pilot"
remote_host="archlinux"
secondary_remote_host=""
remote_ssh_target=""
secondary_remote_ssh_target=""
remote_ssh_user="$(id -un)"
run_id=""
output_root=""
remote_worktree_dir=""
remote_output_dir=""
secondary_remote_worktree_dir=""
secondary_remote_output_dir=""
remote_seed_repo_dir='$HOME/code/psionic'
remote_target_dir='$HOME/.cache/psionic-target/psion-local-first'
remote_tmp_dir='$HOME/.cache/psionic-tmp/psion-local-first'
secondary_remote_target_dir='$HOME/.cache/psionic-target/psion-local-first'
secondary_remote_tmp_dir='$HOME/.cache/psionic-tmp/psion-local-first'
git_ref=""
sync_local_main="0"
allow_local_reference_fallback="0"
dry_run="0"
cleanup_remote="0"
local_tailnet_ip=""
remote_tailnet_ip=""
secondary_remote_tailnet_ip=""
control_plane_host=""
worker_host=""
worker_tailnet_ip=""
worker_count="0"
execution_location=""
execution_topology_classification=""
remote_stage_strategy="not_applicable"
remote_stage_reason=""
secondary_remote_stage_strategy="not_applicable"
secondary_remote_stage_reason=""
max_steps=""
steps_per_window=""
windows_per_cadence=""
step_duration_ms=""
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=5 -C)
scp_opts=(-O -o BatchMode=yes -o ConnectTimeout=5 -C)

usage() {
  cat <<'EOF' >&2
Usage: ./TRAIN [options]

Default behavior:
  - launches the bounded Psion reference-pilot lane, not the actual broader-pretraining lane
  - prefers the accelerator-backed bounded reference pilot
  - stages the current committed git ref to the admitted Tailnet CUDA host
  - runs the accelerated reference pilot there, or the distributed dual-host reference pilot when explicitly requested
  - copies the retained reference-pilot artifacts back to the local machine

Options:
  --workload <reference_pilot|actual_pretraining_bringup>
                                 Workload family. Default: reference_pilot
  --mode <auto|accelerated_reference|local_reference|distributed_reference>
                                 Training mode. Default: auto
  --remote-host <host>           Tailnet SSH target for accelerated runs. Default: archlinux
  --secondary-remote-host <host> Optional second Tailnet SSH target for distributed_reference runs.
  --run-id <id>                  Stable run identifier.
  --output-root <path>           Local retained run root. Default depends on workload.
  --remote-worktree-dir <path>   Remote staged repo root. Default: $HOME/code/psion-reference-pilot/<run_id>/repo
  --remote-output-dir <path>     Remote artifact root. Default: $HOME/code/psion-reference-pilot/<run_id>/output
  --git-ref <ref>                Git ref to stage remotely. Default: local HEAD
  --sync-local-main              Run git pull --ff-only before launch when local checkout is clean.
  --allow-local-reference-fallback
                                 In auto mode, fall back to the bounded CPU reference lane if the remote accelerated lane is unavailable.
  --local-tailnet-ip <ip>        Override local Tailnet IPv4 in the operator manifest.
  --remote-tailnet-ip <ip>       Override remote Tailnet IPv4 in the operator manifest.
  --secondary-remote-tailnet-ip <ip>
                                 Override second remote Tailnet IPv4 in the operator manifest.
  --max-steps <count>            Override the bounded reference-pilot max step count.
  --steps-per-window <count>     Override the bounded reference-pilot steps per logical window.
  --windows-per-cadence <count>  Override the bounded reference-pilot windows per outer cadence.
  --step-duration-ms <ms>        Override the bounded reference-pilot synthetic step duration.
  --cleanup-remote               Remove the staged remote worktree and output after copying artifacts back.
  --dry-run                      Print the bounded reference-pilot plan and write the operator manifest without launching training.
  --help|-h                      Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload)
      workload="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --remote-host)
      remote_host="$2"
      shift 2
      ;;
    --secondary-remote-host)
      secondary_remote_host="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --remote-worktree-dir)
      remote_worktree_dir="$2"
      shift 2
      ;;
    --remote-output-dir)
      remote_output_dir="$2"
      shift 2
      ;;
    --git-ref)
      git_ref="$2"
      shift 2
      ;;
    --sync-local-main)
      sync_local_main="1"
      shift
      ;;
    --allow-local-reference-fallback)
      allow_local_reference_fallback="1"
      shift
      ;;
    --local-tailnet-ip)
      local_tailnet_ip="$2"
      shift 2
      ;;
    --remote-tailnet-ip)
      remote_tailnet_ip="$2"
      shift 2
      ;;
    --secondary-remote-tailnet-ip)
      secondary_remote_tailnet_ip="$2"
      shift 2
      ;;
    --max-steps)
      max_steps="$2"
      shift 2
      ;;
    --steps-per-window)
      steps_per_window="$2"
      shift 2
      ;;
    --windows-per-cadence)
      windows_per_cadence="$2"
      shift 2
      ;;
    --step-duration-ms)
      step_duration_ms="$2"
      shift 2
      ;;
    --cleanup-remote)
      cleanup_remote="1"
      shift
      ;;
    --dry-run)
      dry_run="1"
      shift
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

case "${mode}" in
  auto|accelerated_reference|local_reference|distributed_reference) ;;
  *)
    echo "error: unsupported mode ${mode}" >&2
    usage
    exit 1
    ;;
esac

case "${workload}" in
  reference_pilot|actual_pretraining_bringup) ;;
  *)
    echo "error: unsupported workload ${workload}" >&2
    usage
    exit 1
    ;;
esac

if [[ "${workload}" == "actual_pretraining_bringup" && "${mode}" != "distributed_reference" ]]; then
  echo "error: workload actual_pretraining_bringup currently supports only --mode distributed_reference" >&2
  exit 1
fi

if [[ "${workload}" == "actual_pretraining_bringup" ]]; then
  default_run_prefix="psion-actual-pretraining-bringup"
  default_output_parent="${HOME}/scratch/psion_actual_pretraining_bringup_runs"
  default_remote_root_parent="\$HOME/code/psion-actual-pretraining-bringup"
  operator_manifest_basename="actual_pretraining_bringup_operator_manifest.json"
  summary_basename="actual_pretraining_bringup_operator_summary.json"
  log_basename="actual_pretraining_bringup_train.log"
  artifact_dir_basename="actual_pretraining_bringup_artifacts"
  operator_manifest_schema="psionic.psion_actual_pretraining_bringup_operator_manifest.v1"
  operator_summary_schema="psionic.psion_actual_pretraining_bringup_operator_summary.v1"
  truth_surface_kind="bounded_actual_pretraining_bringup"
  actual_lane_relation="bounded_actual_pretraining_workload"
  workload_claim_boundary="This manifest records one bounded actual-pretraining bringup operator run against the larger internal model and canonical actual dataset identity. It does not claim the full broader long-run actual-pretraining cluster execution."
  distributed_example_name="psion_distributed_actual_pretraining_bringup"
  accelerated_example_name=""
  local_example_name=""
  artifact_prefix="psion_actual_pretraining_bringup"
else
  default_run_prefix="psion-reference-pilot"
  default_output_parent="${HOME}/scratch/psion_reference_pilot_runs"
  default_remote_root_parent="\$HOME/code/psion-reference-pilot"
  operator_manifest_basename="reference_pilot_operator_manifest.json"
  summary_basename="reference_pilot_operator_summary.json"
  log_basename="reference_pilot_train.log"
  artifact_dir_basename="reference_pilot_artifacts"
  operator_manifest_schema="psionic.psion_reference_pilot_operator_manifest.v1"
  operator_summary_schema="psionic.psion_reference_pilot_operator_summary.v1"
  truth_surface_kind="bounded_reference_pilot"
  actual_lane_relation="not_actual_pretraining_lane"
  workload_claim_boundary="This manifest records one bounded Psion reference-pilot operator run. It does not claim the actual broader-pretraining lane. The accelerator-backed mode targets the admitted accelerated reference pilot on the Tailnet CUDA host. The distributed_reference mode targets the bounded multi-host reference lane where the local Apple-silicon host and one or more admitted Tailnet workers contribute optimizer-bearing work. The bounded fallback mode targets the CPU reference pilot only when explicitly allowed."
  distributed_example_name="psion_distributed_reference_pilot"
  accelerated_example_name="psion_accelerated_reference_pilot"
  local_example_name="psion_reference_pilot"
  artifact_prefix="psion_reference_pilot"
fi

now_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if [[ -z "${run_id}" ]]; then
  run_id="${default_run_prefix}-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${output_root}" ]]; then
  output_root="${default_output_parent}/${run_id}"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd)"

if [[ -z "${git_ref}" ]]; then
  git_ref="$(git -C "${repo_root}" rev-parse HEAD)"
fi

if [[ -z "${remote_worktree_dir}" ]]; then
  remote_worktree_dir="${default_remote_root_parent}/${run_id}/repo"
fi
if [[ -z "${remote_output_dir}" ]]; then
  remote_output_dir="${default_remote_root_parent}/${run_id}/output"
fi
if [[ -n "${secondary_remote_host}" && -z "${secondary_remote_worktree_dir}" ]]; then
  secondary_remote_worktree_dir="${default_remote_root_parent}/${run_id}/repo"
fi
if [[ -n "${secondary_remote_host}" && -z "${secondary_remote_output_dir}" ]]; then
  secondary_remote_output_dir="${default_remote_root_parent}/${run_id}/output"
fi

if [[ "${sync_local_main}" == "1" ]]; then
  if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    echo "error: --sync-local-main requires a clean local checkout" >&2
    exit 1
  fi
  if [[ "$(git -C "${repo_root}" rev-parse --abbrev-ref HEAD)" != "main" ]]; then
    echo "error: --sync-local-main requires branch main" >&2
    exit 1
  fi
  git -C "${repo_root}" pull --ff-only
  git_ref="$(git -C "${repo_root}" rev-parse HEAD)"
fi

local_status_branch="$(git -C "${repo_root}" status --short --branch)"
local_dirty="0"
if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  local_dirty="1"
fi

detect_local_tailnet_ip() {
  tailscale_cli ip -4 2>/dev/null | awk 'NF { print; exit }'
}

detect_local_hostname() {
  hostname -s 2>/dev/null || hostname
}

tailscale_cli() {
  if command -v tailscale >/dev/null 2>&1; then
    tailscale "$@"
    return
  fi
  if [[ -x /opt/homebrew/bin/tailscale ]]; then
    /opt/homebrew/bin/tailscale "$@"
    return
  fi
  return 127
}

resolve_tailnet_ipv4() {
  local logical_host="$1"
  tailscale_cli status 2>/dev/null | awk -v host="${logical_host}" '$2 == host { print $1; exit }'
}

resolve_ssh_target() {
  local logical_host="$1"
  if [[ "${logical_host}" == *"@"* ]]; then
    printf '%s\n' "${logical_host}"
    return 0
  fi
  if [[ "${logical_host}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '%s@%s\n' "${remote_ssh_user}" "${logical_host}"
    return 0
  fi
  local resolved_ip=""
  resolved_ip="$(resolve_tailnet_ipv4 "${logical_host}" || true)"
  if [[ -n "${resolved_ip}" ]]; then
    printf '%s@%s\n' "${remote_ssh_user}" "${resolved_ip}"
    return 0
  fi
  printf '%s\n' "${logical_host}"
}

detect_host_tailnet_ip() {
  local logical_host="$1"
  local explicit_ip="$2"
  if [[ -n "${explicit_ip}" ]]; then
    printf '%s\n' "${explicit_ip}"
    return 0
  fi
  if [[ "${logical_host}" == *"@"* ]]; then
    printf '%s\n' "${logical_host##*@}"
    return 0
  fi
  if [[ "${logical_host}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '%s\n' "${logical_host}"
    return 0
  fi
  resolve_tailnet_ipv4 "${logical_host}"
}

remote_preflight_reason=""
remote_gpu_name=""
remote_gpu_busy=""
secondary_remote_preflight_reason=""

remote_accelerated_preflight() {
  if ! ssh "${ssh_opts[@]}" "${remote_ssh_target}" "bash -ic 'command -v cargo >/dev/null && command -v nvidia-smi >/dev/null'" >/dev/null 2>&1; then
    remote_preflight_reason="remote_host_unreachable_or_missing_cargo_or_nvidia_smi"
    return 1
  fi
  remote_gpu_name="$(ssh "${ssh_opts[@]}" "${remote_ssh_target}" "nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1" 2>/dev/null | head -n 1 | tr -d '\r')"
  if [[ -z "${remote_gpu_name}" ]]; then
    remote_preflight_reason="remote_gpu_name_unavailable"
    return 1
  fi
  remote_gpu_busy="$(ssh "${ssh_opts[@]}" "${remote_ssh_target}" "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | awk 'NF { print }'" 2>/dev/null | tr -d '\r')"
  if [[ -n "${remote_gpu_busy}" ]]; then
    remote_preflight_reason="remote_gpu_has_resident_compute_processes"
    return 1
  fi
  return 0
}

secondary_remote_preflight() {
  if [[ -z "${secondary_remote_host}" ]]; then
    secondary_remote_preflight_reason="not_requested"
    return 0
  fi
  if ! ssh "${ssh_opts[@]}" "${secondary_remote_ssh_target}" "bash -ic 'command -v cargo >/dev/null'" >/dev/null 2>&1; then
    secondary_remote_preflight_reason="secondary_remote_unreachable_or_missing_cargo"
    return 1
  fi
  return 0
}

detect_stage_strategy_for_target() {
  local ssh_target="$1"
  if [[ "${selected_mode}" != "accelerated_reference" && "${selected_mode}" != "distributed_reference" ]]; then
    printf 'not_applicable|local_reference_mode\n'
    return 0
  fi
  if ssh "${ssh_opts[@]}" "${ssh_target}" "
    set -euo pipefail
    if [[ ! -d \"${remote_seed_repo_dir}/.git\" ]]; then
      exit 11
    fi
    git -C \"${remote_seed_repo_dir}\" fetch --quiet origin main >/dev/null 2>&1 || true
    git -C \"${remote_seed_repo_dir}\" cat-file -e \"${git_ref}^{commit}\"
  " >/dev/null 2>&1; then
    printf 'remote_git_worktree|remote_seed_repo_contains_git_ref\n'
    return 0
  fi
  if ssh "${ssh_opts[@]}" "${ssh_target}" "test -d \"${remote_seed_repo_dir}/.git\"" >/dev/null 2>&1; then
    printf 'archive_tarball|remote_seed_repo_missing_git_ref\n'
  else
    printf 'archive_tarball|remote_seed_repo_missing\n'
  fi
}

remote_ssh_target="$(resolve_ssh_target "${remote_host}")"
if [[ -n "${secondary_remote_host}" ]]; then
  secondary_remote_ssh_target="$(resolve_ssh_target "${secondary_remote_host}")"
fi

selected_mode="${mode}"
if [[ "${selected_mode}" == "auto" ]]; then
  if remote_accelerated_preflight; then
    selected_mode="accelerated_reference"
  elif [[ "${allow_local_reference_fallback}" == "1" ]]; then
    selected_mode="local_reference"
  else
    echo "error: accelerated Psion lane unavailable: ${remote_preflight_reason}" >&2
    echo "hint: rerun with --allow-local-reference-fallback to use the bounded CPU reference lane instead" >&2
    exit 1
  fi
fi

if [[ "${selected_mode}" == "accelerated_reference" || "${selected_mode}" == "distributed_reference" ]]; then
  if ! remote_accelerated_preflight; then
    echo "error: accelerated Psion lane unavailable: ${remote_preflight_reason}" >&2
    exit 1
  fi
  if [[ "${selected_mode}" == "distributed_reference" && -n "${secondary_remote_host}" ]]; then
    if ! secondary_remote_preflight; then
      echo "error: secondary distributed contributor unavailable: ${secondary_remote_preflight_reason}" >&2
      exit 1
    fi
  fi
fi

if [[ -z "${local_tailnet_ip}" ]]; then
  local_tailnet_ip="$(detect_local_tailnet_ip || true)"
fi
if [[ "${selected_mode}" == "accelerated_reference" || "${selected_mode}" == "distributed_reference" ]] && [[ -z "${remote_tailnet_ip}" ]]; then
  remote_tailnet_ip="$(detect_host_tailnet_ip "${remote_host}" "${remote_tailnet_ip}" || true)"
fi
if [[ "${selected_mode}" == "distributed_reference" && -n "${secondary_remote_host}" && -z "${secondary_remote_tailnet_ip}" ]]; then
  secondary_remote_tailnet_ip="$(detect_host_tailnet_ip "${secondary_remote_host}" "${secondary_remote_tailnet_ip}" || true)"
fi

control_plane_host="$(detect_local_hostname || true)"
if [[ -z "${control_plane_host}" ]]; then
  control_plane_host="unknown"
fi

if [[ "${selected_mode}" == "accelerated_reference" ]]; then
  worker_host="${remote_host}"
  worker_tailnet_ip="${remote_tailnet_ip}"
  worker_count="1"
  execution_location="remote"
  execution_topology_classification="local_control_plane_single_remote_worker"
elif [[ "${selected_mode}" == "distributed_reference" ]]; then
  worker_host="${remote_host}"
  worker_tailnet_ip="${remote_tailnet_ip}"
  if [[ -n "${secondary_remote_host}" ]]; then
    worker_count="3"
    execution_topology_classification="multi_host_joint_gradient_average"
  else
    worker_count="2"
    execution_topology_classification="dual_host_joint_gradient_average"
  fi
  execution_location="hybrid_cluster"
else
  worker_host="${control_plane_host}"
  worker_tailnet_ip="${local_tailnet_ip}"
  worker_count="1"
  execution_location="local"
  execution_topology_classification="single_local_control_plane_worker"
fi

IFS='|' read -r remote_stage_strategy remote_stage_reason < <(detect_stage_strategy_for_target "${remote_ssh_target}")
if [[ -n "${secondary_remote_host}" ]]; then
  IFS='|' read -r secondary_remote_stage_strategy secondary_remote_stage_reason < <(detect_stage_strategy_for_target "${secondary_remote_ssh_target}")
fi

operator_manifest_path="${output_root}/${operator_manifest_basename}"
summary_path="${output_root}/${summary_basename}"
local_log_path="${output_root}/${log_basename}"
local_artifact_dir="${output_root}/${artifact_dir_basename}"

python3 - <<'PY' \
  "${operator_manifest_path}" "${run_id}" "${workload}" "${truth_surface_kind}" "${actual_lane_relation}" "${operator_manifest_schema}" "${workload_claim_boundary}" "${artifact_prefix}" "${selected_mode}" "${git_ref}" \
  "${remote_host}" "${output_root}" "${remote_worktree_dir}" "${remote_output_dir}" \
  "${secondary_remote_host}" "${secondary_remote_worktree_dir}" "${secondary_remote_output_dir}" \
  "${local_tailnet_ip}" "${remote_tailnet_ip}" "${secondary_remote_tailnet_ip}" "${local_dirty}" "${local_status_branch}" \
  "${remote_gpu_name}" "${remote_preflight_reason}" "${secondary_remote_preflight_reason}" "${control_plane_host}" \
  "${worker_host}" "${worker_tailnet_ip}" "${worker_count}" \
  "${execution_location}" "${execution_topology_classification}" \
  "${remote_stage_strategy}" "${remote_stage_reason}" "${secondary_remote_stage_strategy}" "${secondary_remote_stage_reason}" \
  "${max_steps}" "${steps_per_window}" "${windows_per_cadence}" "${step_duration_ms}"
import json
import sys
from datetime import datetime

(
    path,
    run_id,
    workload,
    truth_surface_kind,
    actual_lane_relation,
    schema_version,
    claim_boundary,
    artifact_prefix,
    selected_mode,
    git_ref,
    remote_host,
    output_root,
    remote_worktree_dir,
    remote_output_dir,
    secondary_remote_host,
    secondary_remote_worktree_dir,
    secondary_remote_output_dir,
    local_tailnet_ip,
    remote_tailnet_ip,
    secondary_remote_tailnet_ip,
    local_dirty,
    local_status_branch,
    remote_gpu_name,
    remote_preflight_reason,
    secondary_remote_preflight_reason,
    control_plane_host,
    worker_host,
    worker_tailnet_ip,
    worker_count,
    execution_location,
    execution_topology_classification,
    remote_stage_strategy,
    remote_stage_reason,
    secondary_remote_stage_strategy,
    secondary_remote_stage_reason,
    max_steps,
    steps_per_window,
    windows_per_cadence,
    step_duration_ms,
) = sys.argv[1:]

doc = {
    "schema_version": schema_version,
    "created_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "run_id": run_id,
    "workload": workload,
    "truth_surface_kind": truth_surface_kind,
    "actual_lane_relation": actual_lane_relation,
    "artifact_prefix": artifact_prefix,
    "selected_mode": selected_mode,
    "git_ref": git_ref,
    "local_output_root": output_root,
    "local_repo_dirty": local_dirty == "1",
    "local_status_branch": local_status_branch,
    "control_plane_host": control_plane_host or None,
    "control_plane_tailnet_ip": local_tailnet_ip or None,
    "local_tailnet_ip": local_tailnet_ip or None,
    "worker_host": worker_host or None,
    "worker_tailnet_ip": worker_tailnet_ip or None,
    "worker_count": int(worker_count),
    "worker_hosts": [
        value
        for value in [control_plane_host or None, remote_host or None, secondary_remote_host or None]
        if value is not None and value != ""
    ],
    "execution_location": execution_location,
    "execution_topology_classification": execution_topology_classification,
    "remote_host": remote_host,
    "remote_tailnet_ip": remote_tailnet_ip or None,
    "secondary_remote_host": secondary_remote_host or None,
    "secondary_remote_tailnet_ip": secondary_remote_tailnet_ip or None,
    "remote_gpu_name": remote_gpu_name or None,
    "remote_preflight_reason": remote_preflight_reason or None,
    "secondary_remote_preflight_reason": secondary_remote_preflight_reason or None,
    "remote_stage_strategy": remote_stage_strategy,
    "remote_stage_reason": remote_stage_reason or None,
    "remote_worktree_dir": remote_worktree_dir,
    "remote_output_dir": remote_output_dir,
    "secondary_remote_stage_strategy": secondary_remote_stage_strategy or None,
    "secondary_remote_stage_reason": secondary_remote_stage_reason or None,
    "secondary_remote_worktree_dir": secondary_remote_worktree_dir or None,
    "secondary_remote_output_dir": secondary_remote_output_dir or None,
    "requested_budget_override": {
        "max_steps": int(max_steps) if max_steps else None,
        "steps_per_window": int(steps_per_window) if steps_per_window else None,
        "windows_per_cadence": int(windows_per_cadence) if windows_per_cadence else None,
        "step_duration_ms": int(step_duration_ms) if step_duration_ms else None,
    },
    "claim_boundary": claim_boundary,
}

with open(path, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
PY

if [[ "${dry_run}" == "1" ]]; then
  echo "status=dry_run"
  echo "workload=${workload}"
  echo "run_id=${run_id}"
  echo "selected_mode=${selected_mode}"
  echo "git_ref=${git_ref}"
  echo "output_root=${output_root}"
  echo "control_plane_host=${control_plane_host}"
  echo "worker_host=${worker_host}"
  echo "worker_count=${worker_count}"
  echo "execution_location=${execution_location}"
  echo "max_steps=${max_steps:-default}"
  echo "steps_per_window=${steps_per_window:-default}"
  echo "windows_per_cadence=${windows_per_cadence:-default}"
  echo "step_duration_ms=${step_duration_ms:-default}"
  echo "operator_manifest=${operator_manifest_path}"
  if [[ "${selected_mode}" == "accelerated_reference" || "${selected_mode}" == "distributed_reference" ]]; then
    echo "remote_host=${remote_host}"
    echo "remote_tailnet_ip=${remote_tailnet_ip}"
    echo "remote_gpu_name=${remote_gpu_name}"
    echo "remote_stage_strategy=${remote_stage_strategy}"
    if [[ "${selected_mode}" == "distributed_reference" ]]; then
      echo "distributed_topology=${execution_topology_classification}"
      if [[ -n "${secondary_remote_host}" ]]; then
        echo "secondary_remote_host=${secondary_remote_host}"
        echo "secondary_remote_tailnet_ip=${secondary_remote_tailnet_ip}"
        echo "secondary_remote_stage_strategy=${secondary_remote_stage_strategy}"
      fi
    fi
  fi
  exit 0
fi

write_summary() {
  python3 - <<'PY' \
    "${summary_path}" "${operator_manifest_path}" "${operator_summary_schema}" "${workload}" "${truth_surface_kind}" "${actual_lane_relation}" "${artifact_prefix}" "$@"
import hashlib
import json
import os
import sys
from datetime import datetime

summary_path = sys.argv[1]
manifest_path = sys.argv[2]
schema_version = sys.argv[3]
workload = sys.argv[4]
truth_surface_kind = sys.argv[5]
actual_lane_relation = sys.argv[6]
artifact_prefix = sys.argv[7]
mode = sys.argv[8]
status = sys.argv[9]
output_root = sys.argv[10]
log_path = sys.argv[11]
artifact_dir = sys.argv[12]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

summary = {
    "schema_version": schema_version,
    "recorded_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "workload": workload,
    "truth_surface_kind": truth_surface_kind,
    "actual_lane_relation": actual_lane_relation,
    "artifact_prefix": artifact_prefix,
    "operator_manifest_path": os.path.abspath(manifest_path),
    "operator_manifest_sha256": sha256_file(manifest_path),
    "selected_mode": mode,
    "status": status,
    "output_root": os.path.abspath(output_root),
    "log_path": os.path.abspath(log_path) if log_path else None,
    "artifact_dir": os.path.abspath(artifact_dir) if artifact_dir else None,
}

for key in [
    "run_id",
    "git_ref",
    "control_plane_host",
    "control_plane_tailnet_ip",
    "worker_host",
    "worker_tailnet_ip",
    "worker_count",
    "worker_hosts",
    "execution_location",
    "execution_topology_classification",
    "remote_host",
    "remote_tailnet_ip",
    "secondary_remote_host",
    "secondary_remote_tailnet_ip",
    "remote_gpu_name",
    "remote_stage_strategy",
    "remote_stage_reason",
    "secondary_remote_stage_strategy",
    "secondary_remote_stage_reason",
]:
    if key in manifest:
        summary[key] = manifest[key]

budget_override = manifest.get("requested_budget_override")
if budget_override:
    summary["requested_budget_override"] = budget_override

stage_prefix = artifact_prefix
stage_path = os.path.join(artifact_dir, f"{stage_prefix}_stage_receipt.json")
observability_path = os.path.join(artifact_dir, f"{stage_prefix}_observability_receipt.json")
checkpoint_path = os.path.join(artifact_dir, f"{stage_prefix}_checkpoint_manifest.json")
cluster_topology_path = os.path.join(artifact_dir, f"{stage_prefix}_cluster_topology_receipt.json")
cluster_step_path = os.path.join(artifact_dir, f"{stage_prefix}_cluster_step_receipts.json")
cluster_contribution_path = os.path.join(artifact_dir, f"{stage_prefix}_cluster_contribution_receipts.json")
cluster_progress_checkpoint_path = os.path.join(artifact_dir, f"{stage_prefix}_cluster_progress_checkpoint_receipts.json")
cluster_contributor_continuity_path = os.path.join(artifact_dir, f"{stage_prefix}_cluster_contributor_continuity_receipt.json")
dual_host_topology_path = os.path.join(artifact_dir, f"{stage_prefix}_dual_host_topology_receipt.json")
dual_host_step_path = os.path.join(artifact_dir, f"{stage_prefix}_dual_host_step_receipts.json")
dual_host_contribution_path = os.path.join(artifact_dir, f"{stage_prefix}_dual_host_contribution_receipts.json")
progress_checkpoint_path = os.path.join(artifact_dir, f"{stage_prefix}_progress_checkpoint_receipts.json")
contributor_continuity_path = os.path.join(artifact_dir, f"{stage_prefix}_contributor_continuity_receipt.json")
progress_checkpoint_dir = os.path.join(artifact_dir, f"{stage_prefix}_progress_checkpoints")

def attach_common_run_surfaces():
    for key, path in [
        ("stage_receipt_path", stage_path),
        ("observability_receipt_path", observability_path),
        ("checkpoint_manifest_path", checkpoint_path),
    ]:
        summary[key] = os.path.abspath(path)
        if os.path.exists(path):
            summary[f"{key}_sha256"] = sha256_file(path)
    if os.path.exists(stage_path):
        with open(stage_path, "r", encoding="utf-8") as handle:
            stage = json.load(handle)
        delivered = stage.get("delivered_execution") or {}
        summary["stage_id"] = stage.get("stage_id")
        summary["delivered_backend"] = delivered.get("runtime_backend")
        summary["delivered_devices"] = delivered.get("selected_devices")
        summary["stage_receipt_digest"] = stage.get("receipt_digest")
    if os.path.exists(observability_path):
        with open(observability_path, "r", encoding="utf-8") as handle:
            observability = json.load(handle)
        cost = observability.get("cost") or {}
        hardware = observability.get("hardware_topology") or {}
        summary["observability_receipt_digest"] = observability.get("observability_digest")
        summary["cost"] = cost
        summary["total_cost_microusd"] = cost.get("total_cost_microusd")
        if hardware:
            summary["observed_worker_count"] = hardware.get("observed_worker_count")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        summary["checkpoint_ref"] = checkpoint.get("checkpoint_ref")
        summary["checkpoint_parameter_state_digest"] = checkpoint.get("parameter_state_digest")
        summary["model_id"] = checkpoint.get("model_id")
        summary["dataset_identity"] = checkpoint.get("dataset_identity")
    selected_progress_path = cluster_progress_checkpoint_path if os.path.exists(cluster_progress_checkpoint_path) else progress_checkpoint_path
    selected_continuity_path = cluster_contributor_continuity_path if os.path.exists(cluster_contributor_continuity_path) else contributor_continuity_path
    if os.path.exists(selected_progress_path):
        summary["progress_checkpoint_receipts_path"] = os.path.abspath(selected_progress_path)
        summary["progress_checkpoint_receipts_path_sha256"] = sha256_file(selected_progress_path)
        with open(selected_progress_path, "r", encoding="utf-8") as handle:
            progress_receipts = json.load(handle)
        summary["progress_checkpoint_count"] = len(progress_receipts)
        if progress_receipts:
            summary["progress_window_count"] = len({receipt.get("window_index") for receipt in progress_receipts})
            summary["progress_cadence_count"] = len({receipt.get("cadence_index") for receipt in progress_receipts if receipt.get("is_cadence_boundary")})
            summary["final_cumulative_train_tokens_processed"] = progress_receipts[-1].get("cumulative_train_tokens_processed")
            summary["final_cumulative_mean_tokens_per_second"] = progress_receipts[-1].get("cumulative_mean_tokens_per_second")
    if os.path.exists(selected_continuity_path):
        summary["contributor_continuity_receipt_path"] = os.path.abspath(selected_continuity_path)
        summary["contributor_continuity_receipt_path_sha256"] = sha256_file(selected_continuity_path)
        with open(selected_continuity_path, "r", encoding="utf-8") as handle:
            continuity = json.load(handle)
        summary["all_configured_contributors_present_each_step"] = continuity.get("all_configured_contributors_present_each_step")
    if os.path.isdir(progress_checkpoint_dir):
        summary["progress_checkpoint_directory"] = os.path.abspath(progress_checkpoint_dir)

if mode == "accelerated_reference" and status == "completed":
    stage_path = os.path.join(artifact_dir, "psion_accelerated_reference_pilot_stage_receipt.json")
    observability_path = os.path.join(artifact_dir, "psion_accelerated_reference_pilot_observability_receipt.json")
    checkpoint_path = os.path.join(artifact_dir, "psion_accelerated_reference_pilot_checkpoint_manifest.json")
    for key, path in [
        ("stage_receipt_path", stage_path),
        ("observability_receipt_path", observability_path),
        ("checkpoint_manifest_path", checkpoint_path),
    ]:
        summary[key] = os.path.abspath(path)
        if os.path.exists(path):
            summary[f"{key}_sha256"] = sha256_file(path)
    if os.path.exists(stage_path):
        with open(stage_path, "r", encoding="utf-8") as handle:
            stage = json.load(handle)
        delivered = stage.get("delivered_execution") or {}
        summary["delivered_backend"] = delivered.get("runtime_backend")
        summary["delivered_devices"] = delivered.get("selected_devices")
        summary["stage_receipt_digest"] = stage.get("receipt_digest")
    if os.path.exists(observability_path):
        with open(observability_path, "r", encoding="utf-8") as handle:
            observability = json.load(handle)
        cost = observability.get("cost") or {}
        summary["observability_receipt_digest"] = observability.get("observability_digest")
        summary["cost"] = cost
        summary["total_cost_microusd"] = cost.get("total_cost_microusd")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        summary["checkpoint_ref"] = checkpoint.get("checkpoint_ref")
        summary["checkpoint_parameter_state_digest"] = checkpoint.get("parameter_state_digest")
elif mode == "distributed_reference" and status == "completed":
    attach_common_run_surfaces()
    topology_path = cluster_topology_path if os.path.exists(cluster_topology_path) else dual_host_topology_path
    step_path = cluster_step_path if os.path.exists(cluster_step_path) else dual_host_step_path
    contribution_path = cluster_contribution_path if os.path.exists(cluster_contribution_path) else dual_host_contribution_path
    topology_key = "cluster_topology_receipt_path" if os.path.exists(cluster_topology_path) else "dual_host_topology_receipt_path"
    step_key = "cluster_step_receipts_path" if os.path.exists(cluster_step_path) else "dual_host_step_receipts_path"
    contribution_key = "cluster_contribution_receipts_path" if os.path.exists(cluster_contribution_path) else "dual_host_contribution_receipts_path"
    for key, path in [
        (topology_key, topology_path),
        (step_key, step_path),
        (contribution_key, contribution_path),
    ]:
        summary[key] = os.path.abspath(path)
        if os.path.exists(path):
            summary[f"{key}_sha256"] = sha256_file(path)
    if os.path.exists(topology_path):
        with open(topology_path, "r", encoding="utf-8") as handle:
            topology = json.load(handle)
        summary["execution_topology_classification"] = topology.get("execution_topology_classification")
        summary["local_runtime_backend"] = topology.get("local_runtime_backend")
        summary["remote_runtime_backend"] = topology.get("remote_runtime_backend")
        summary["remote_worker_host"] = topology.get("remote_worker_host")
        summary["secondary_remote_worker_host"] = topology.get("secondary_remote_worker_host")
        summary["worker_hosts"] = topology.get("worker_hosts")
        summary["runtime_backends"] = topology.get("runtime_backends")
        summary["contributor_count"] = topology.get("contributor_count")
elif mode == "local_reference" and status == "completed":
    attach_common_run_surfaces()

with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
PY
}

mkdir -p "${local_artifact_dir}"
: > "${local_log_path}"

log_note() {
  printf '[%s] %s\n' "$(now_utc)" "$*" >>"${local_log_path}"
}

run_with_reference_pilot_env() {
  if (( ${#reference_pilot_env[@]} )); then
    env "${reference_pilot_env[@]}" "$@"
  else
    env "$@"
  fi
}

reference_pilot_env=()
if [[ -n "${max_steps}" ]]; then
  reference_pilot_env+=("PSION_REFERENCE_PILOT_MAX_STEPS=${max_steps}")
fi
if [[ -n "${steps_per_window}" ]]; then
  reference_pilot_env+=("PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=${steps_per_window}")
fi
if [[ -n "${windows_per_cadence}" ]]; then
  reference_pilot_env+=("PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=${windows_per_cadence}")
fi
if [[ -n "${step_duration_ms}" ]]; then
  reference_pilot_env+=("PSION_REFERENCE_PILOT_STEP_DURATION_MS=${step_duration_ms}")
fi

stage_remote_repo() {
  local ssh_target="$1"
  local stage_strategy="$2"
  local worktree_dir="$3"
  local output_dir="$4"
  if [[ "${stage_strategy}" == "remote_git_worktree" ]]; then
    log_note "staging_strategy=remote_git_worktree ssh_target=${ssh_target} git_ref=${git_ref} remote_seed_repo_dir=${remote_seed_repo_dir}"
    ssh "${ssh_opts[@]}" "${ssh_target}" "
      set -euo pipefail
      rm -rf \"${output_dir}\"
      mkdir -p \"\$(dirname \"${worktree_dir}\")\" \"${output_dir}\"
      git -C \"${remote_seed_repo_dir}\" fetch --quiet origin main >/dev/null 2>&1 || true
      if git -C \"${remote_seed_repo_dir}\" worktree list --porcelain | grep -Fqx \"worktree ${worktree_dir}\"; then
        git -C \"${remote_seed_repo_dir}\" worktree remove --force \"${worktree_dir}\"
      else
        rm -rf \"${worktree_dir}\"
      fi
      git -C \"${remote_seed_repo_dir}\" worktree add --detach --force \"${worktree_dir}\" \"${git_ref}\"
    " >>"${local_log_path}" 2>&1
    return 0
  fi

  local local_stage_archive remote_stage_archive
  local_stage_archive="$(mktemp "${output_root}/stage-${run_id}.XXXXXX.tar.gz")"
  remote_stage_archive="${worktree_dir}.tar.gz"
  log_note "staging_strategy=archive_tarball ssh_target=${ssh_target} git_ref=${git_ref} local_stage_archive=${local_stage_archive}"
  git -C "${repo_root}" archive --format=tar.gz -o "${local_stage_archive}" "${git_ref}" \
    >>"${local_log_path}" 2>&1
  ssh "${ssh_opts[@]}" "${ssh_target}" "
    set -euo pipefail
    mkdir -p \"\$(dirname \"${worktree_dir}\")\"
  " >>"${local_log_path}" 2>&1
  scp "${scp_opts[@]}" "${local_stage_archive}" "${ssh_target}:${remote_stage_archive}" \
    >>"${local_log_path}" 2>&1
  ssh "${ssh_opts[@]}" "${ssh_target}" "
    set -euo pipefail
    rm -rf \"${worktree_dir}\" \"${output_dir}\"
    mkdir -p \"${worktree_dir}\" \"${output_dir}\"
    tar -xzf \"${remote_stage_archive}\" -C \"${worktree_dir}\"
    rm -f \"${remote_stage_archive}\"
  " >>"${local_log_path}" 2>&1
  rm -f "${local_stage_archive}"
}

cleanup_remote_target() {
  local ssh_target="$1"
  local stage_strategy="$2"
  local worktree_dir="$3"
  local output_dir="$4"
  if [[ "${stage_strategy}" == "remote_git_worktree" ]]; then
    ssh "${ssh_opts[@]}" "${ssh_target}" "
      set -euo pipefail
      git -C \"${remote_seed_repo_dir}\" worktree remove --force \"${worktree_dir}\"
      rm -rf \"${output_dir}\"
      worktree_parent=\"\$(dirname \"${worktree_dir}\")\"
      output_parent=\"\$(dirname \"${output_dir}\")\"
      rmdir \"\${worktree_parent}\" 2>/dev/null || true
      if [[ \"\${output_parent}\" != \"\${worktree_parent}\" ]]; then
        rmdir \"\${output_parent}\" 2>/dev/null || true
      fi
    " >>"${local_log_path}" 2>&1
  else
    ssh "${ssh_opts[@]}" "${ssh_target}" "
      set -euo pipefail
      rm -rf \"${worktree_dir}\" \"${output_dir}\"
      worktree_parent=\"\$(dirname \"${worktree_dir}\")\"
      output_parent=\"\$(dirname \"${output_dir}\")\"
      rmdir \"\${worktree_parent}\" 2>/dev/null || true
      if [[ \"\${output_parent}\" != \"\${worktree_parent}\" ]]; then
        rmdir \"\${output_parent}\" 2>/dev/null || true
      fi
    " >>"${local_log_path}" 2>&1
  fi
}

on_exit() {
  local exit_code="$1"
  if [[ "${exit_code}" -ne 0 ]] && [[ ! -f "${summary_path}" ]]; then
    write_summary "${selected_mode}" "failed" "${output_root}" "${local_log_path}" "${local_artifact_dir}" || true
  fi
}
trap 'on_exit $?' EXIT

if [[ "${selected_mode}" == "local_reference" ]]; then
  log_note "launching workload=${workload} mode=local_reference control_plane_host=${control_plane_host}"
  run_with_reference_pilot_env cargo run -q -p psionic-train --example "${local_example_name}" -- "${local_artifact_dir}" \
    >>"${local_log_path}" 2>&1
  write_summary "${selected_mode}" "completed" "${output_root}" "${local_log_path}" "${local_artifact_dir}"
  echo "status=completed"
  echo "workload=${workload}"
  echo "mode=${selected_mode}"
  echo "run_id=${run_id}"
  echo "output_root=${output_root}"
  echo "artifact_dir=${local_artifact_dir}"
  echo "operator_manifest=${operator_manifest_path}"
  echo "operator_summary=${summary_path}"
  trap - EXIT
  exit 0
fi

log_note "launching workload=${workload} mode=${selected_mode} control_plane_host=${control_plane_host} worker_host=${worker_host} worker_count=${worker_count}"
stage_remote_repo "${remote_ssh_target}" "${remote_stage_strategy}" "${remote_worktree_dir}" "${remote_output_dir}"
if [[ "${selected_mode}" == "distributed_reference" && -n "${secondary_remote_host}" ]]; then
  stage_remote_repo "${secondary_remote_ssh_target}" "${secondary_remote_stage_strategy}" "${secondary_remote_worktree_dir}" "${secondary_remote_output_dir}"
fi

remote_reference_pilot_exports=""
if [[ -n "${max_steps}" ]]; then
  remote_reference_pilot_exports="${remote_reference_pilot_exports} export PSION_REFERENCE_PILOT_MAX_STEPS=${max_steps};"
fi
if [[ -n "${steps_per_window}" ]]; then
  remote_reference_pilot_exports="${remote_reference_pilot_exports} export PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=${steps_per_window};"
fi
if [[ -n "${windows_per_cadence}" ]]; then
  remote_reference_pilot_exports="${remote_reference_pilot_exports} export PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=${windows_per_cadence};"
fi
if [[ -n "${step_duration_ms}" ]]; then
  remote_reference_pilot_exports="${remote_reference_pilot_exports} export PSION_REFERENCE_PILOT_STEP_DURATION_MS=${step_duration_ms};"
fi

if [[ "${selected_mode}" == "accelerated_reference" ]]; then
  remote_command="bash -ic 'export CARGO_TARGET_DIR=${remote_target_dir}; export TMPDIR=${remote_tmp_dir}; export RUST_MIN_STACK=16777216;${remote_reference_pilot_exports} mkdir -p \"${remote_target_dir}\" \"${remote_tmp_dir}\"; cd ${remote_worktree_dir} && cargo run -q -p psionic-train --example ${accelerated_example_name} -- ${remote_output_dir}'"
  ssh "${ssh_opts[@]}" "${remote_ssh_target}" "${remote_command}" >>"${local_log_path}" 2>&1

  rm -rf "${local_artifact_dir}"
  mkdir -p "${local_artifact_dir}"
  log_note "copying_retained_artifacts remote_output_dir=${remote_output_dir}"
  ssh "${ssh_opts[@]}" "${remote_ssh_target}" "
    set -euo pipefail
    tar -cf - -C \"${remote_output_dir}\" .
  " 2>>"${local_log_path}" | tar -xf - -C "${local_artifact_dir}" >>"${local_log_path}" 2>&1
else
  rm -rf "${local_artifact_dir}"
  mkdir -p "${local_artifact_dir}"
  log_note "launching workload=${workload} mode=distributed_reference control_plane_host=${control_plane_host} remote_worker_host=${remote_host} secondary_remote_worker_host=${secondary_remote_host:-none}"
  run_with_reference_pilot_env \
    "PSION_REFERENCE_PILOT_REMOTE_SSH_TARGET=${remote_ssh_target}" \
    "PSION_REFERENCE_PILOT_REMOTE_WORKTREE_DIR=${remote_worktree_dir}" \
    "PSION_REFERENCE_PILOT_REMOTE_OUTPUT_DIR=${remote_output_dir}" \
    "PSION_REFERENCE_PILOT_REMOTE_TARGET_DIR=${remote_target_dir}" \
    "PSION_REFERENCE_PILOT_REMOTE_TMP_DIR=${remote_tmp_dir}" \
    "PSION_REFERENCE_PILOT_CONTROL_PLANE_HOST=${control_plane_host}" \
    "PSION_REFERENCE_PILOT_CONTROL_PLANE_TAILNET_IP=${local_tailnet_ip}" \
    "PSION_REFERENCE_PILOT_REMOTE_HOST=${remote_host}" \
    "PSION_REFERENCE_PILOT_REMOTE_TAILNET_IP=${remote_tailnet_ip}" \
    "PSION_REFERENCE_PILOT_DUAL_HOST_REMOTE_BACKEND=cuda" \
    "PSION_REFERENCE_PILOT_SECONDARY_REMOTE_HOST=${secondary_remote_host}" \
    "PSION_REFERENCE_PILOT_SECONDARY_REMOTE_TAILNET_IP=${secondary_remote_tailnet_ip}" \
    "PSION_REFERENCE_PILOT_SECONDARY_REMOTE_SSH_TARGET=${secondary_remote_ssh_target}" \
    "PSION_REFERENCE_PILOT_SECONDARY_REMOTE_WORKTREE_DIR=${secondary_remote_worktree_dir}" \
    "PSION_REFERENCE_PILOT_SECONDARY_REMOTE_OUTPUT_DIR=${secondary_remote_output_dir}" \
    "PSION_REFERENCE_PILOT_SECONDARY_REMOTE_TARGET_DIR=${secondary_remote_target_dir}" \
    "PSION_REFERENCE_PILOT_SECONDARY_REMOTE_TMP_DIR=${secondary_remote_tmp_dir}" \
    "PSION_REFERENCE_PILOT_DUAL_HOST_SECONDARY_REMOTE_BACKEND=cpu" \
    cargo run -q -p psionic-train --example "${distributed_example_name}" -- "${local_artifact_dir}" \
    >>"${local_log_path}" 2>&1
fi

if [[ "${cleanup_remote}" == "1" ]]; then
  log_note "cleanup_remote=1 remote_stage_strategy=${remote_stage_strategy}"
  cleanup_remote_target "${remote_ssh_target}" "${remote_stage_strategy}" "${remote_worktree_dir}" "${remote_output_dir}"
  if [[ "${selected_mode}" == "distributed_reference" && -n "${secondary_remote_host}" ]]; then
    log_note "cleanup_secondary_remote=1 secondary_remote_stage_strategy=${secondary_remote_stage_strategy}"
    cleanup_remote_target "${secondary_remote_ssh_target}" "${secondary_remote_stage_strategy}" "${secondary_remote_worktree_dir}" "${secondary_remote_output_dir}"
  fi
fi

write_summary "${selected_mode}" "completed" "${output_root}" "${local_log_path}" "${local_artifact_dir}"

echo "status=completed"
echo "workload=${workload}"
echo "mode=${selected_mode}"
echo "run_id=${run_id}"
echo "output_root=${output_root}"
echo "artifact_dir=${local_artifact_dir}"
echo "log_path=${local_log_path}"
echo "operator_manifest=${operator_manifest_path}"
echo "operator_summary=${summary_path}"
trap - EXIT
