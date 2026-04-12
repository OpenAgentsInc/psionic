#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

mode="auto"
remote_host="archlinux"
remote_ssh_target=""
remote_ssh_user="$(id -un)"
run_id=""
output_root=""
remote_worktree_dir=""
remote_output_dir=""
remote_seed_repo_dir='$HOME/code/psionic'
remote_target_dir='$HOME/.cache/psionic-target/psion-local-first'
remote_tmp_dir='$HOME/.cache/psionic-tmp/psion-local-first'
git_ref=""
sync_local_main="0"
allow_local_reference_fallback="0"
dry_run="0"
cleanup_remote="0"
local_tailnet_ip=""
remote_tailnet_ip=""
control_plane_host=""
worker_host=""
worker_tailnet_ip=""
worker_count="0"
execution_location=""
execution_topology_classification=""
remote_stage_strategy="not_applicable"
remote_stage_reason=""
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=5)
scp_opts=(-O -o BatchMode=yes -o ConnectTimeout=5)

usage() {
  cat <<'EOF' >&2
Usage: ./TRAIN [options]

Default behavior:
  - launches the bounded Psion reference-pilot lane, not the actual broader-pretraining lane
  - prefers the accelerator-backed bounded reference pilot
  - stages the current committed git ref to the admitted Tailnet CUDA host
  - runs the accelerated reference pilot there
  - copies the retained reference-pilot artifacts back to the local machine

Options:
  --mode <auto|accelerated_reference|local_reference>
                                 Training mode. Default: auto
  --remote-host <host>           Tailnet SSH target for accelerated runs. Default: archlinux
  --run-id <id>                  Stable run identifier.
  --output-root <path>           Local reference-pilot run root. Default: ~/scratch/psion_reference_pilot_runs/<run_id>
  --remote-worktree-dir <path>   Remote staged repo root. Default: $HOME/code/psion-reference-pilot/<run_id>/repo
  --remote-output-dir <path>     Remote artifact root. Default: $HOME/code/psion-reference-pilot/<run_id>/output
  --git-ref <ref>                Git ref to stage remotely. Default: local HEAD
  --sync-local-main              Run git pull --ff-only before launch when local checkout is clean.
  --allow-local-reference-fallback
                                 In auto mode, fall back to the bounded CPU reference lane if the remote accelerated lane is unavailable.
  --local-tailnet-ip <ip>        Override local Tailnet IPv4 in the operator manifest.
  --remote-tailnet-ip <ip>       Override remote Tailnet IPv4 in the operator manifest.
  --cleanup-remote               Remove the staged remote worktree and output after copying artifacts back.
  --dry-run                      Print the bounded reference-pilot plan and write the operator manifest without launching training.
  --help|-h                      Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --remote-host)
      remote_host="$2"
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
  auto|accelerated_reference|local_reference) ;;
  *)
    echo "error: unsupported mode ${mode}" >&2
    usage
    exit 1
    ;;
esac

now_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if [[ -z "${run_id}" ]]; then
  run_id="psion-reference-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${output_root}" ]]; then
  output_root="${HOME}/scratch/psion_reference_pilot_runs/${run_id}"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd)"

if [[ -z "${git_ref}" ]]; then
  git_ref="$(git -C "${repo_root}" rev-parse HEAD)"
fi

if [[ -z "${remote_worktree_dir}" ]]; then
  remote_worktree_dir="\$HOME/code/psion-reference-pilot/${run_id}/repo"
fi
if [[ -z "${remote_output_dir}" ]]; then
  remote_output_dir="\$HOME/code/psion-reference-pilot/${run_id}/output"
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

resolve_remote_ssh_target() {
  if [[ "${remote_host}" == *"@"* ]]; then
    printf '%s\n' "${remote_host}"
    return 0
  fi
  if [[ "${remote_host}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '%s@%s\n' "${remote_ssh_user}" "${remote_host}"
    return 0
  fi
  local resolved_ip=""
  resolved_ip="$(resolve_tailnet_ipv4 "${remote_host}" || true)"
  if [[ -n "${resolved_ip}" ]]; then
    printf '%s@%s\n' "${remote_ssh_user}" "${resolved_ip}"
    return 0
  fi
  printf '%s\n' "${remote_host}"
}

detect_remote_tailnet_ip() {
  if [[ -n "${remote_tailnet_ip}" ]]; then
    printf '%s\n' "${remote_tailnet_ip}"
    return 0
  fi
  if [[ "${remote_host}" == *"@"* ]]; then
    printf '%s\n' "${remote_host##*@}"
    return 0
  fi
  if [[ "${remote_host}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '%s\n' "${remote_host}"
    return 0
  fi
  resolve_tailnet_ipv4 "${remote_host}"
}

remote_preflight_reason=""
remote_gpu_name=""
remote_gpu_busy=""

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

detect_remote_stage_strategy() {
  remote_stage_strategy="archive_tarball"
  remote_stage_reason="remote_seed_repo_missing_git_ref"
  if [[ "${selected_mode}" != "accelerated_reference" ]]; then
    remote_stage_strategy="not_applicable"
    remote_stage_reason="local_reference_mode"
    return 0
  fi
  if ssh "${ssh_opts[@]}" "${remote_ssh_target}" "
    set -euo pipefail
    if [[ ! -d \"${remote_seed_repo_dir}/.git\" ]]; then
      exit 11
    fi
    git -C \"${remote_seed_repo_dir}\" fetch --quiet origin main >/dev/null 2>&1 || true
    git -C \"${remote_seed_repo_dir}\" cat-file -e \"${git_ref}^{commit}\"
  " >/dev/null 2>&1; then
    remote_stage_strategy="remote_git_worktree"
    remote_stage_reason="remote_seed_repo_contains_git_ref"
    return 0
  fi
  if ssh "${ssh_opts[@]}" "${remote_ssh_target}" "test -d \"${remote_seed_repo_dir}/.git\"" >/dev/null 2>&1; then
    remote_stage_reason="remote_seed_repo_missing_git_ref"
  else
    remote_stage_reason="remote_seed_repo_missing"
  fi
}

remote_ssh_target="$(resolve_remote_ssh_target)"

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

if [[ "${selected_mode}" == "accelerated_reference" ]]; then
  if ! remote_accelerated_preflight; then
    echo "error: accelerated Psion lane unavailable: ${remote_preflight_reason}" >&2
    exit 1
  fi
fi

if [[ -z "${local_tailnet_ip}" ]]; then
  local_tailnet_ip="$(detect_local_tailnet_ip || true)"
fi
if [[ "${selected_mode}" == "accelerated_reference" ]] && [[ -z "${remote_tailnet_ip}" ]]; then
  remote_tailnet_ip="$(detect_remote_tailnet_ip || true)"
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
else
  worker_host="${control_plane_host}"
  worker_tailnet_ip="${local_tailnet_ip}"
  worker_count="1"
  execution_location="local"
  execution_topology_classification="single_local_control_plane_worker"
fi

detect_remote_stage_strategy

operator_manifest_path="${output_root}/reference_pilot_operator_manifest.json"
summary_path="${output_root}/reference_pilot_operator_summary.json"
local_log_path="${output_root}/reference_pilot_train.log"
local_artifact_dir="${output_root}/reference_pilot_artifacts"

python3 - <<'PY' \
  "${operator_manifest_path}" "${run_id}" "${selected_mode}" "${git_ref}" \
  "${remote_host}" "${output_root}" "${remote_worktree_dir}" "${remote_output_dir}" \
  "${local_tailnet_ip}" "${remote_tailnet_ip}" "${local_dirty}" "${local_status_branch}" \
  "${remote_gpu_name}" "${remote_preflight_reason}" "${control_plane_host}" \
  "${worker_host}" "${worker_tailnet_ip}" "${worker_count}" \
  "${execution_location}" "${execution_topology_classification}" \
  "${remote_stage_strategy}" "${remote_stage_reason}"
import json
import sys
from datetime import datetime

(
    path,
    run_id,
    selected_mode,
    git_ref,
    remote_host,
    output_root,
    remote_worktree_dir,
    remote_output_dir,
    local_tailnet_ip,
    remote_tailnet_ip,
    local_dirty,
    local_status_branch,
    remote_gpu_name,
    remote_preflight_reason,
    control_plane_host,
    worker_host,
    worker_tailnet_ip,
    worker_count,
    execution_location,
    execution_topology_classification,
    remote_stage_strategy,
    remote_stage_reason,
) = sys.argv[1:]

doc = {
    "schema_version": "psionic.psion_reference_pilot_operator_manifest.v1",
    "created_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "run_id": run_id,
    "truth_surface_kind": "bounded_reference_pilot",
    "actual_lane_relation": "not_actual_pretraining_lane",
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
    "execution_location": execution_location,
    "execution_topology_classification": execution_topology_classification,
    "remote_host": remote_host,
    "remote_tailnet_ip": remote_tailnet_ip or None,
    "remote_gpu_name": remote_gpu_name or None,
    "remote_preflight_reason": remote_preflight_reason or None,
    "remote_stage_strategy": remote_stage_strategy,
    "remote_stage_reason": remote_stage_reason or None,
    "remote_worktree_dir": remote_worktree_dir,
    "remote_output_dir": remote_output_dir,
    "claim_boundary": "This manifest records one bounded Psion reference-pilot operator run. It does not claim the actual broader-pretraining lane. The accelerator-backed mode targets the admitted accelerated reference pilot on the Tailnet CUDA host. The bounded fallback mode targets the CPU reference pilot only when explicitly allowed.",
}

with open(path, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
PY

if [[ "${dry_run}" == "1" ]]; then
  echo "status=dry_run"
  echo "run_id=${run_id}"
  echo "selected_mode=${selected_mode}"
  echo "git_ref=${git_ref}"
  echo "output_root=${output_root}"
  echo "control_plane_host=${control_plane_host}"
  echo "worker_host=${worker_host}"
  echo "worker_count=${worker_count}"
  echo "execution_location=${execution_location}"
  echo "reference_pilot_operator_manifest=${operator_manifest_path}"
  if [[ "${selected_mode}" == "accelerated_reference" ]]; then
    echo "remote_host=${remote_host}"
    echo "remote_tailnet_ip=${remote_tailnet_ip}"
    echo "remote_gpu_name=${remote_gpu_name}"
    echo "remote_stage_strategy=${remote_stage_strategy}"
  fi
  exit 0
fi

write_summary() {
  python3 - <<'PY' \
    "${summary_path}" "${operator_manifest_path}" "$@"
import hashlib
import json
import os
import sys
from datetime import datetime

summary_path = sys.argv[1]
manifest_path = sys.argv[2]
mode = sys.argv[3]
status = sys.argv[4]
output_root = sys.argv[5]
log_path = sys.argv[6]
artifact_dir = sys.argv[7]

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
    "schema_version": "psionic.psion_reference_pilot_operator_summary.v1",
    "recorded_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "truth_surface_kind": "bounded_reference_pilot",
    "actual_lane_relation": "not_actual_pretraining_lane",
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
    "execution_location",
    "execution_topology_classification",
    "remote_host",
    "remote_tailnet_ip",
    "remote_gpu_name",
    "remote_stage_strategy",
    "remote_stage_reason",
]:
    if key in manifest:
        summary[key] = manifest[key]

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
elif mode == "local_reference" and status == "completed":
    stage_path = os.path.join(artifact_dir, "psion_reference_pilot_stage_receipt.json")
    observability_path = os.path.join(artifact_dir, "psion_reference_pilot_observability_receipt.json")
    checkpoint_path = os.path.join(artifact_dir, "psion_reference_pilot_checkpoint_manifest.json")
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

stage_remote_repo() {
  if [[ "${remote_stage_strategy}" == "remote_git_worktree" ]]; then
    log_note "staging_strategy=remote_git_worktree git_ref=${git_ref} remote_seed_repo_dir=${remote_seed_repo_dir}"
    ssh "${ssh_opts[@]}" "${remote_ssh_target}" "
      set -euo pipefail
      rm -rf \"${remote_output_dir}\"
      mkdir -p \"\$(dirname \"${remote_worktree_dir}\")\" \"${remote_output_dir}\"
      git -C \"${remote_seed_repo_dir}\" fetch --quiet origin main >/dev/null 2>&1 || true
      if git -C \"${remote_seed_repo_dir}\" worktree list --porcelain | grep -Fqx \"worktree ${remote_worktree_dir}\"; then
        git -C \"${remote_seed_repo_dir}\" worktree remove --force \"${remote_worktree_dir}\"
      else
        rm -rf \"${remote_worktree_dir}\"
      fi
      git -C \"${remote_seed_repo_dir}\" worktree add --detach --force \"${remote_worktree_dir}\" \"${git_ref}\"
    " >>"${local_log_path}" 2>&1
    return 0
  fi

  local local_stage_archive remote_stage_archive
  local_stage_archive="$(mktemp "${output_root}/stage-${run_id}.XXXXXX.tar")"
  remote_stage_archive="${remote_worktree_dir}.tar"
  log_note "staging_strategy=archive_tarball git_ref=${git_ref} local_stage_archive=${local_stage_archive}"
  git -C "${repo_root}" archive --format=tar -o "${local_stage_archive}" "${git_ref}" \
    >>"${local_log_path}" 2>&1
  scp "${scp_opts[@]}" "${local_stage_archive}" "${remote_ssh_target}:${remote_stage_archive}" \
    >>"${local_log_path}" 2>&1
  ssh "${ssh_opts[@]}" "${remote_ssh_target}" "
    set -euo pipefail
    rm -rf \"${remote_worktree_dir}\" \"${remote_output_dir}\"
    mkdir -p \"${remote_worktree_dir}\" \"${remote_output_dir}\"
    tar -xf \"${remote_stage_archive}\" -C \"${remote_worktree_dir}\"
    rm -f \"${remote_stage_archive}\"
  " >>"${local_log_path}" 2>&1
  rm -f "${local_stage_archive}"
}

on_exit() {
  local exit_code="$1"
  if [[ "${exit_code}" -ne 0 ]] && [[ ! -f "${summary_path}" ]]; then
    write_summary "${selected_mode}" "failed" "${output_root}" "${local_log_path}" "${local_artifact_dir}" || true
  fi
}
trap 'on_exit $?' EXIT

if [[ "${selected_mode}" == "local_reference" ]]; then
  log_note "launching local_reference control_plane_host=${control_plane_host}"
  cargo run -q -p psionic-train --example psion_reference_pilot -- "${local_artifact_dir}" \
    >>"${local_log_path}" 2>&1
  write_summary "${selected_mode}" "completed" "${output_root}" "${local_log_path}" "${local_artifact_dir}"
  echo "status=completed"
  echo "mode=${selected_mode}"
  echo "run_id=${run_id}"
  echo "output_root=${output_root}"
  echo "reference_pilot_artifact_dir=${local_artifact_dir}"
  echo "reference_pilot_operator_manifest=${operator_manifest_path}"
  echo "reference_pilot_operator_summary=${summary_path}"
  trap - EXIT
  exit 0
fi

log_note "launching accelerated_reference control_plane_host=${control_plane_host} worker_host=${worker_host} worker_count=${worker_count}"
stage_remote_repo

remote_command="bash -ic 'export CARGO_TARGET_DIR=${remote_target_dir}; export TMPDIR=${remote_tmp_dir}; mkdir -p \"${remote_target_dir}\" \"${remote_tmp_dir}\"; cd ${remote_worktree_dir} && cargo run -q -p psionic-train --example psion_accelerated_reference_pilot -- ${remote_output_dir}'"
ssh "${ssh_opts[@]}" "${remote_ssh_target}" "${remote_command}" >>"${local_log_path}" 2>&1

rm -rf "${local_artifact_dir}"
mkdir -p "${local_artifact_dir}"
log_note "copying_retained_artifacts remote_output_dir=${remote_output_dir}"
ssh "${ssh_opts[@]}" "${remote_ssh_target}" "
  set -euo pipefail
  tar -cf - -C \"${remote_output_dir}\" .
" 2>>"${local_log_path}" | tar -xf - -C "${local_artifact_dir}" >>"${local_log_path}" 2>&1

if [[ "${cleanup_remote}" == "1" ]]; then
  log_note "cleanup_remote=1 remote_stage_strategy=${remote_stage_strategy}"
  if [[ "${remote_stage_strategy}" == "remote_git_worktree" ]]; then
    ssh "${ssh_opts[@]}" "${remote_ssh_target}" "
      set -euo pipefail
      git -C \"${remote_seed_repo_dir}\" worktree remove --force \"${remote_worktree_dir}\"
      rm -rf \"${remote_output_dir}\"
    " >>"${local_log_path}" 2>&1
  else
    ssh "${ssh_opts[@]}" "${remote_ssh_target}" "
      set -euo pipefail
      rm -rf \"${remote_worktree_dir}\" \"${remote_output_dir}\"
    " >>"${local_log_path}" 2>&1
  fi
fi

write_summary "${selected_mode}" "completed" "${output_root}" "${local_log_path}" "${local_artifact_dir}"

echo "status=completed"
echo "mode=${selected_mode}"
echo "run_id=${run_id}"
echo "output_root=${output_root}"
echo "reference_pilot_artifact_dir=${local_artifact_dir}"
echo "log_path=${local_log_path}"
echo "reference_pilot_operator_manifest=${operator_manifest_path}"
echo "reference_pilot_operator_summary=${summary_path}"
trap - EXIT
