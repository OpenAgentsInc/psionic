#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

run_id=""
bundle_dir=""
manifest_only=false
mac_host_alias="swarm-mac-a.local"
linux_host_alias="swarm-linux-4080-a.local"
mac_repo_dir="~/code/psionic"
linux_repo_dir="~/code/psionic"
mac_run_root=""
linux_run_root=""

usage() {
  cat <<'EOF' >&2
Usage: first-swarm-launch-trusted-lan.sh [options]

Options:
  --run-id <run_id>              Stable run identifier. Default: first-swarm-trusted-lan-<utc>
  --bundle-dir <path>            Output directory for the operator bundle.
  --mac-host-alias <host>        Host alias for the Mac coordinator. Default: swarm-mac-a.local
  --linux-host-alias <host>      Host alias for the Linux contributor. Default: swarm-linux-4080-a.local
  --mac-repo-dir <path>          Repo checkout on the Mac host. Default: ~/code/psionic
  --linux-repo-dir <path>        Repo checkout on the Linux host. Default: ~/code/psionic
  --mac-run-root <path>          Run root on the Mac host. Default: ~/swarm-runs/<run_id>/mac
  --linux-run-root <path>        Run root on the Linux host. Default: ~/swarm-runs/<run_id>/linux
  --manifest-only                Materialize the local operator bundle and stop before any remote action.
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
    --mac-host-alias)
      mac_host_alias="$2"
      shift 2
      ;;
    --linux-host-alias)
      linux_host_alias="$2"
      shift 2
      ;;
    --mac-repo-dir)
      mac_repo_dir="$2"
      shift 2
      ;;
    --linux-repo-dir)
      linux_repo_dir="$2"
      shift 2
      ;;
    --mac-run-root)
      mac_run_root="$2"
      shift 2
      ;;
    --linux-run-root)
      linux_run_root="$2"
      shift 2
      ;;
    --manifest-only)
      manifest_only=true
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

now_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

now_ms() {
  python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
}

file_sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

append_phase_result() {
  local phase_id="$1"
  local status="$2"
  local started_at_ms="$3"
  local finished_at_ms="$4"
  local log_path="$5"
  local artifact_path="$6"
  phase_results="$(
    jq -c \
      --arg phase_id "${phase_id}" \
      --arg status "${status}" \
      --argjson started_at_ms "${started_at_ms}" \
      --argjson finished_at_ms "${finished_at_ms}" \
      --arg log_path "${log_path}" \
      --arg artifact_path "${artifact_path}" \
      '. + [{
        phase_id: $phase_id,
        status: $status,
        started_at_ms: $started_at_ms,
        finished_at_ms: $finished_at_ms,
        log_path: $log_path,
        artifact_path: $artifact_path
      }]' <<<"${phase_results}"
  )"
}

run_phase() {
  local phase_id="$1"
  local artifact_path="$2"
  shift 2
  local log_path="${logs_dir}/${phase_id}.log"
  local started_at_ms finished_at_ms
  started_at_ms="$(now_ms)"
  if "$@" >"${log_path}" 2>&1; then
    finished_at_ms="$(now_ms)"
    append_phase_result "${phase_id}" "completed" "${started_at_ms}" "${finished_at_ms}" "${log_path}" "${artifact_path}"
  else
    finished_at_ms="$(now_ms)"
    append_phase_result "${phase_id}" "failed" "${started_at_ms}" "${finished_at_ms}" "${log_path}" "${artifact_path}"
    echo "error: phase ${phase_id} failed; see ${log_path}" >&2
    exit 1
  fi
}

if [[ -z "${run_id}" ]]; then
  run_id="first-swarm-trusted-lan-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${bundle_dir}" ]]; then
  bundle_dir="${repo_root}/tmp/${run_id}"
fi

if [[ -z "${mac_run_root}" ]]; then
  mac_run_root="~/swarm-runs/${run_id}/mac"
fi

if [[ -z "${linux_run_root}" ]]; then
  linux_run_root="~/swarm-runs/${run_id}/linux"
fi

reports_dir="${bundle_dir}/reports"
logs_dir="${bundle_dir}/logs"
mkdir -p "${reports_dir}" "${logs_dir}"

topology_path="${bundle_dir}/first_swarm_trusted_lan_topology_contract_v1.json"
failure_drills_path="${reports_dir}/first_swarm_trusted_lan_failure_drills_v1.json"
workflow_plan_path="${bundle_dir}/first_swarm_live_workflow_plan_v1.json"
mac_bringup_path="${reports_dir}/swarm_mac_mlx_bringup_v1.json"
linux_bringup_path="${reports_dir}/swarm_linux_rtx4080_bringup_v1.json"
manifest_path="${bundle_dir}/first_swarm_trusted_lan_launch_manifest.json"
receipt_path="${bundle_dir}/first_swarm_trusted_lan_launch_receipt.json"

phase_results='[]'

run_phase \
  "topology_contract" \
  "${topology_path}" \
  cargo run -q -p psionic-train --bin first_swarm_trusted_lan_topology_contract -- "${topology_path}"

run_phase \
  "failure_drills" \
  "${failure_drills_path}" \
  cargo run -q -p psionic-train --bin first_swarm_trusted_lan_failure_drills -- "${failure_drills_path}"

run_phase \
  "workflow_plan" \
  "${workflow_plan_path}" \
  cargo run -q -p psionic-mlx-workflows --bin first_swarm_live_workflow_plan -- "${workflow_plan_path}"

run_phase \
  "copy_mac_bringup_fixture" \
  "${mac_bringup_path}" \
  cp "${repo_root}/fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json" "${mac_bringup_path}"

run_phase \
  "copy_linux_bringup_fixture" \
  "${linux_bringup_path}" \
  cp "${repo_root}/fixtures/swarm/reports/swarm_linux_rtx4080_bringup_v1.json" "${linux_bringup_path}"

topology_digest="$(jq -r '.contract_digest' "${topology_path}")"
failure_drills_digest="$(jq -r '.bundle_digest' "${failure_drills_path}")"
workflow_plan_digest="$(jq -r '.plan_digest' "${workflow_plan_path}")"
workflow_membership_receipt_digest="$(jq -r '.membership_receipt.receipt_digest' "${workflow_plan_path}")"
mac_bringup_digest="$(jq -r '.report_digest' "${mac_bringup_path}")"
linux_bringup_digest="$(jq -r '.report_digest' "${linux_bringup_path}")"
swarm_contract_digest="$(jq -r '.swarm_contract_digest' "${topology_path}")"
receipt_contract_digest="$(jq -r '.receipt_contract_digest' "${topology_path}")"

mac_bringup_command="cd \"${mac_repo_dir}\" && scripts/check-swarm-mac-mlx-bringup.sh --report \"${mac_run_root}/reports/swarm_mac_mlx_bringup_v1.json\""
linux_bringup_command="cd \"${linux_repo_dir}\" && scripts/check-swarm-linux-4080-bringup.sh --report \"${linux_run_root}/reports/swarm_linux_rtx4080_bringup_v1.json\""
workflow_plan_command="cd \"${mac_repo_dir}\" && cargo run -q -p psionic-mlx-workflows --bin first_swarm_live_workflow_plan -- \"${bundle_dir}/first_swarm_live_workflow_plan_v1.json\""
topology_contract_command="cd \"${mac_repo_dir}\" && cargo run -q -p psionic-train --bin first_swarm_trusted_lan_topology_contract -- \"${bundle_dir}/first_swarm_trusted_lan_topology_contract_v1.json\""
failure_drills_command="cd \"${mac_repo_dir}\" && cargo run -q -p psionic-train --bin first_swarm_trusted_lan_failure_drills -- \"${bundle_dir}/reports/first_swarm_trusted_lan_failure_drills_v1.json\""

manifest_json="$(
  jq -n \
    --arg schema_version "swarm.first_trusted_lan_launch_manifest.v1" \
    --arg created_at_utc "$(now_utc)" \
    --arg run_id "${run_id}" \
    --arg run_family_id "$(jq -r '.run_family_id' "${topology_path}")" \
    --arg swarm_contract_digest "${swarm_contract_digest}" \
    --arg receipt_contract_digest "${receipt_contract_digest}" \
    --arg topology_contract_path "${topology_path}" \
    --arg topology_contract_digest "${topology_digest}" \
    --arg failure_drills_path "${failure_drills_path}" \
    --arg failure_drills_digest "${failure_drills_digest}" \
    --arg workflow_plan_path "${workflow_plan_path}" \
    --arg workflow_plan_digest "${workflow_plan_digest}" \
    --arg workflow_membership_receipt_digest "${workflow_membership_receipt_digest}" \
    --arg bundle_dir "${bundle_dir}" \
    --arg manifest_path "${manifest_path}" \
    --arg receipt_path "${receipt_path}" \
    --arg mac_host_alias "${mac_host_alias}" \
    --arg linux_host_alias "${linux_host_alias}" \
    --arg mac_repo_dir "${mac_repo_dir}" \
    --arg linux_repo_dir "${linux_repo_dir}" \
    --arg mac_run_root "${mac_run_root}" \
    --arg linux_run_root "${linux_run_root}" \
    --arg mac_bringup_fixture_path "${mac_bringup_path}" \
    --arg mac_bringup_digest "${mac_bringup_digest}" \
    --arg linux_bringup_fixture_path "${linux_bringup_path}" \
    --arg linux_bringup_digest "${linux_bringup_digest}" \
    --arg mac_bringup_command "${mac_bringup_command}" \
    --arg linux_bringup_command "${linux_bringup_command}" \
    --arg workflow_plan_command "${workflow_plan_command}" \
    --arg topology_contract_command "${topology_contract_command}" \
    --arg failure_drills_command "${failure_drills_command}" \
    --arg manifest_only "${manifest_only}" \
    --argjson launch_sequence "$(jq -c '.launch_sequence' "${topology_path}")" \
    --argjson phase_results "${phase_results}" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      run_id: $run_id,
      run_family_id: $run_family_id,
      launcher: {
        manifest_only: ($manifest_only == "true"),
        execution_posture: "local_bundle_materialization_only"
      },
      contract_digests: {
        swarm_contract_digest: $swarm_contract_digest,
        receipt_contract_digest: $receipt_contract_digest,
        topology_contract_digest: $topology_contract_digest,
        failure_drills_digest: $failure_drills_digest,
        workflow_plan_digest: $workflow_plan_digest,
        workflow_membership_receipt_digest: $workflow_membership_receipt_digest
      },
      retained_paths: {
        bundle_dir: $bundle_dir,
        topology_contract_path: $topology_contract_path,
        failure_drills_path: $failure_drills_path,
        workflow_plan_path: $workflow_plan_path,
        mac_bringup_fixture_path: $mac_bringup_fixture_path,
        linux_bringup_fixture_path: $linux_bringup_fixture_path,
        manifest_path: $manifest_path,
        receipt_path: $receipt_path
      },
      hosts: {
        mac: {
          host_alias: $mac_host_alias,
          repo_dir: $mac_repo_dir,
          run_root: $mac_run_root,
          expected_bringup_report_digest: $mac_bringup_digest
        },
        linux: {
          host_alias: $linux_host_alias,
          repo_dir: $linux_repo_dir,
          run_root: $linux_run_root,
          expected_bringup_report_digest: $linux_bringup_digest
        }
      },
      remote_commands: {
        mac_bringup_command: $mac_bringup_command,
        linux_bringup_command: $linux_bringup_command,
        workflow_plan_command: $workflow_plan_command,
        topology_contract_command: $topology_contract_command,
        failure_drills_command: $failure_drills_command
      },
      launch_sequence: $launch_sequence,
      local_phase_results: $phase_results,
      claim_boundary: "This launcher materializes the exact first-swarm trusted-LAN operator bundle and the pinned per-host commands. It does not claim remote execution, training success, or promotion by itself."
    }'
)"

printf '%s\n' "${manifest_json}" > "${manifest_path}"
manifest_digest="$(file_sha256 "${manifest_path}")"

receipt_json="$(
  jq -n \
    --arg schema_version "swarm.first_trusted_lan_launch_receipt.v1" \
    --arg created_at_utc "$(now_utc)" \
    --arg run_id "${run_id}" \
    --arg manifest_path "${manifest_path}" \
    --arg manifest_digest "${manifest_digest}" \
    --arg receipt_path "${receipt_path}" \
    --argjson phase_results "${phase_results}" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      run_id: $run_id,
      launch_status: "bundle_materialized",
      manifest_path: $manifest_path,
      manifest_digest: $manifest_digest,
      receipt_path: $receipt_path,
      phase_results: $phase_results
    }'
)"

printf '%s\n' "${receipt_json}" > "${receipt_path}"
cat "${manifest_path}"
