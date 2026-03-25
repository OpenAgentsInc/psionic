#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

pod_host=""
pod_port=""
ssh_user="root"
ssh_key_path=""
run_id=""
remote_root="/workspace/parameter-golf-runpod"
input_target_root="/workspace/parameter-golf"
local_bundle_root=""
allow_small_root=0
minimum_remote_free_gib=25

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-launch-current-bundle-8xh100.sh [options]

Options:
  --pod-host <host>           RunPod pod host or IP.
  --pod-port <port>           SSH port for the pod.
  --ssh-user <user>           SSH user. Default: root.
  --ssh-key <path>            SSH private key path.
  --run-id <run_id>           Stable run identifier. Default: timestamped.
  --remote-root <path>        Remote run-root parent. Default: /workspace/parameter-golf-runpod.
  --input-target-root <path>  Remote public cache target root. Default: /workspace/parameter-golf.
  --local-bundle-root <path>  Reuse an existing local bundle root instead of rebuilding one.
  --allow-small-root          Permit launch onto a root that fails the free-space guard.
  --minimum-remote-free-gib <n>
                              Minimum free GiB required on the selected remote root unless
                              --allow-small-root is set. Default: 25.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pod-host)
      pod_host="$2"
      shift 2
      ;;
    --pod-port)
      pod_port="$2"
      shift 2
      ;;
    --ssh-user)
      ssh_user="$2"
      shift 2
      ;;
    --ssh-key)
      ssh_key_path="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --remote-root)
      remote_root="$2"
      shift 2
      ;;
    --input-target-root)
      input_target_root="$2"
      shift 2
      ;;
    --local-bundle-root)
      local_bundle_root="$2"
      shift 2
      ;;
    --allow-small-root)
      allow_small_root=1
      shift 1
      ;;
    --minimum-remote-free-gib)
      minimum_remote_free_gib="$2"
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

if [[ -z "${pod_host}" || -z "${pod_port}" ]]; then
  echo "error: --pod-host and --pod-port are required" >&2
  usage
  exit 1
fi

if [[ -z "${run_id}" ]]; then
  run_id="parameter-golf-runpod-scoreproof-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if ! [[ "${minimum_remote_free_gib}" =~ ^[0-9]+$ ]] || [[ "${minimum_remote_free_gib}" -eq 0 ]]; then
  echo "error: --minimum-remote-free-gib must be one positive integer" >&2
  exit 1
fi

if [[ -z "${local_bundle_root}" ]]; then
  local_bundle_root="/tmp/psionic-pgolf-current-bundle-${run_id}"
  local_submission_dir="${local_bundle_root}/exported_submission/records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2"
  rm -rf "${local_bundle_root}"
  mkdir -p "${local_bundle_root}"
  cargo run -q -p psionic-train --example parameter_golf_non_record_submission_bundle -- "${local_submission_dir}"
else
  local_submission_dir="${local_bundle_root}/exported_submission/records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2"
fi

runtime_path="${local_submission_dir}/runtime/parameter_golf_submission_runtime"
if [[ ! -f "${runtime_path}" ]]; then
  echo "error: expected runtime at ${runtime_path}" >&2
  exit 1
fi
runtime_sha="$(sha256sum "${runtime_path}" | awk '{print $1}')"
local_git_ref="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"

ssh_cmd=(ssh -o StrictHostKeyChecking=no -p "${pod_port}")
if [[ -n "${ssh_key_path}" ]]; then
  ssh_cmd+=(-i "${ssh_key_path}")
fi
ssh_cmd+=("${ssh_user}@${pod_host}")

run_root="${remote_root}/${run_id}"
remote_submission_dir="${run_root}/exported_submission/records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2"
remote_support_root="${run_root}/support"
remote_receipt_path="${run_root}/parameter_golf_runpod_8xh100_current_bundle_launch.json"

"${ssh_cmd[@]}" "rm -rf \"${run_root}\" && mkdir -p \"${run_root}\" \"${remote_support_root}\""
tar -C "${local_bundle_root}" -cf - exported_submission | "${ssh_cmd[@]}" "tar --no-same-owner --no-same-permissions -xf - -C \"${run_root}\""
tar -C "${repo_root}" -cf - \
  scripts/parameter-golf-materialize-public-cache.sh \
  scripts/parameter-golf-read-input-materialization-env.sh \
  fixtures/parameter_golf/google/parameter_golf_google_input_contract_v1.json \
  | "${ssh_cmd[@]}" "tar --no-same-owner --no-same-permissions -xf - -C \"${remote_support_root}\""

"${ssh_cmd[@]}" bash -s -- \
  "${run_id}" \
  "${run_root}" \
  "${remote_submission_dir}" \
  "${remote_support_root}" \
  "${input_target_root}" \
  "${runtime_sha}" \
  "${local_git_ref}" \
  "${remote_receipt_path}" \
  "${allow_small_root}" \
  "${minimum_remote_free_gib}" <<'REMOTE'
set -euo pipefail

run_id="$1"
run_root="$2"
submission_dir="$3"
support_root="$4"
input_target_root="$5"
runtime_sha="$6"
local_git_ref="$7"
receipt_path="$8"
allow_small_root="$9"
minimum_remote_free_gib="${10}"

free_kib="$(df -Pk "${run_root}" | awk 'NR==2 {print $4}')"
mount_target="$(df -Pk "${run_root}" | awk 'NR==2 {print $6}')"
free_bytes="$((free_kib * 1024))"
minimum_free_bytes="$((minimum_remote_free_gib * 1024 * 1024 * 1024))"
if [[ "${allow_small_root}" != "1" && "${free_bytes}" -lt "${minimum_free_bytes}" ]]; then
  echo "error: selected run root ${run_root} is on mount ${mount_target} with ${free_bytes} free bytes; require at least ${minimum_free_bytes}. pass --allow-small-root to override." >&2
  exit 1
fi

bash "${support_root}/scripts/parameter-golf-materialize-public-cache.sh" \
  --contract "${support_root}/fixtures/parameter_golf/google/parameter_golf_google_input_contract_v1.json" \
  --target-root "${input_target_root}" \
  > "${run_root}/parameter_golf_input_materialization.json"

eval "$(
  bash "${support_root}/scripts/parameter-golf-read-input-materialization-env.sh" \
    --report "${run_root}/parameter_golf_input_materialization.json"
)"

cd "${submission_dir}"
remote_runtime_sha="$(sha256sum runtime/parameter_golf_submission_runtime | awk '{print $1}')"
if [[ "${remote_runtime_sha}" != "${runtime_sha}" ]]; then
  echo "error: remote runtime digest ${remote_runtime_sha} did not match local ${runtime_sha}" >&2
  exit 1
fi

nohup env \
  PSIONIC_PARAMETER_GOLF_EXECUTION_MODE=distributed_8xh100_train \
  WORLD_SIZE=8 \
  NCCL_DEBUG=WARN \
  python3 train_gpt.py > "${run_root}/execution.log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "${pid}" > "${run_root}/execution.pid"

cat > "${receipt_path}" <<EOF
{
  "schema_version": "parameter_golf.runpod_8xh100_current_bundle_launch.v1",
  "run_id": "${run_id}",
  "run_root": "${run_root}",
  "submission_dir": "${submission_dir}",
  "input_target_root": "${input_target_root}",
  "runtime_sha256": "${runtime_sha}",
  "local_git_ref": "${local_git_ref}",
  "execution_pid": ${pid},
  "filesystem_mount_point": "${mount_target}",
  "filesystem_free_bytes_before_launch": ${free_bytes},
  "minimum_required_free_bytes": ${minimum_free_bytes},
  "small_root_override": $( [[ "${allow_small_root}" == "1" ]] && printf 'true' || printf 'false' )
}
EOF

printf 'run_id=%s\nrun_root=%s\nsubmission_dir=%s\nruntime_sha256=%s\nexecution_pid=%s\n' \
  "${run_id}" \
  "${run_root}" \
  "${submission_dir}" \
  "${runtime_sha}" \
  "${pid}"
REMOTE
