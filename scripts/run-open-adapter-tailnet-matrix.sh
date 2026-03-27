#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

run_id=""
bundle_dir=""
target_seconds="600"
batch_size="8"
git_ref=""
remote_host="archlinux"
remote_worktree_dir="/tmp/psionic-tailrun-matrix"
remote_output_root=""
remote_cargo_home="/tmp/psionic-tailrun-cargo-home"
remote_target_root="/tmp/psionic-tailrun-target"

usage() {
  cat <<'EOF' >&2
Usage: scripts/run-open-adapter-tailnet-matrix.sh [options]

Options:
  --run-id <id>                 Stable run identifier.
  --bundle-dir <path>           Local artifact root.
  --target-seconds <seconds>    Shared benchmark wallclock. Default: 600
  --batch-size <size>           PGOLF-ish training batch size. Default: 8
  --git-ref <ref>               Git ref used for the remote worktree. Default: local HEAD
  --remote-host <host>          Remote admitted host. Default: archlinux
  --remote-worktree-dir <path>  Remote clean worktree. Default: /tmp/psionic-tailrun-matrix
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
    --target-seconds)
      target_seconds="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --git-ref)
      git_ref="$2"
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

if [[ -z "${run_id}" ]]; then
  run_id="tailrun-admitted-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${bundle_dir}" ]]; then
  bundle_dir="${repo_root}/fixtures/apple_adapter/runs/${run_id}"
fi

if [[ -z "${git_ref}" ]]; then
  git_ref="$(git -C "${repo_root}" rev-parse HEAD)"
fi

remote_output_root="/tmp/${run_id}-${remote_host}"
mkdir -p "${bundle_dir}"
bundle_dir="$(cd "${bundle_dir}" && pwd)"

local_dir="${bundle_dir}/m5_mlx"
remote_dir="${bundle_dir}/archlinux_cuda"
log_dir="${bundle_dir}/logs"
matrix_report_path="${bundle_dir}/matrix_report.json"

mkdir -p "${local_dir}" "${remote_dir}" "${log_dir}"

remote_log_path="${log_dir}/remote.log"
local_log_path="${log_dir}/local.log"

echo "Preparing remote worktree ${remote_worktree_dir} on ${remote_host}"
git -C "${repo_root}" archive "${git_ref}" | ssh "${remote_host}" "
  set -euo pipefail
  rm -rf ${remote_worktree_dir}
  mkdir -p ${remote_worktree_dir}
  tar -xf - -C ${remote_worktree_dir}
  mkdir -p ${remote_cargo_home} ${remote_target_root}
  if [[ -d ~/.cargo/registry ]]; then
    ln -sfn ~/.cargo/registry ${remote_cargo_home}/registry
  fi
  if [[ -d ~/.cargo/git ]]; then
    ln -sfn ~/.cargo/git ${remote_cargo_home}/git
  fi
"

echo "Starting remote admitted-device benchmark on ${remote_host}"
ssh "${remote_host}" "bash -ic 'export CARGO_HOME=${remote_cargo_home}; export CARGO_TARGET_DIR=${remote_target_root}/${run_id}; cd ${remote_worktree_dir} && cargo build -q -p psionic-train --bin open_adapter_same_node_wallclock_benchmark && \${CARGO_TARGET_DIR}/debug/open_adapter_same_node_wallclock_benchmark --backend-label open_adapter_backend.cuda.gpt_oss_lm_head --output-root ${remote_output_root} --target-seconds ${target_seconds} --batch-size ${batch_size}'" \
  >"${remote_log_path}" 2>&1 &
remote_pid=$!

cleanup() {
  if [[ -n "${remote_pid:-}" ]] && kill -0 "${remote_pid}" >/dev/null 2>&1; then
    kill "${remote_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Running local admitted-device benchmark"
(
  cd "${repo_root}"
  cargo build -q -p psionic-train --bin open_adapter_same_node_wallclock_benchmark
  target/debug/open_adapter_same_node_wallclock_benchmark \
    --backend-label open_adapter_backend.mlx.metal.gpt_oss_lm_head \
    --output-root "${local_dir}" \
    --target-seconds "${target_seconds}" \
    --batch-size "${batch_size}"
) >"${local_log_path}" 2>&1

wait "${remote_pid}"
trap - EXIT

scp "${remote_host}:${remote_output_root}/report.json" "${remote_dir}/report.json" >/dev/null
scp "${remote_host}:${remote_output_root}/portable_bundle.safetensors" "${remote_dir}/portable_bundle.safetensors" >/dev/null

jq -n \
  --arg schema_version "psionic.open_adapter_tailnet_admitted_device_matrix.v1" \
  --arg run_id "${run_id}" \
  --arg git_ref "${git_ref}" \
  --arg remote_host "${remote_host}" \
  --argjson target_seconds "${target_seconds}" \
  --argjson batch_size "${batch_size}" \
  --slurpfile local_report "${local_dir}/report.json" \
  --slurpfile remote_report "${remote_dir}/report.json" \
  '
  def ratio(a; b): if b == 0 then null else a / b end;
  def pct_gain(a; b): if b == 0 then null else ((a - b) / b) * 100 end;
  {
    schema_version: $schema_version,
    run_id: $run_id,
    git_ref: $git_ref,
    target_wallclock_seconds: $target_seconds,
    batch_size: $batch_size,
    admitted_device_set: [
      {
        host: $local_report[0].host,
        backend_label: $local_report[0].backend_label,
        report_path: "m5_mlx/report.json",
        bundle_path: "m5_mlx/portable_bundle.safetensors"
      },
      {
        host: $remote_host,
        backend_label: $remote_report[0].backend_label,
        report_path: "archlinux_cuda/report.json",
        bundle_path: "archlinux_cuda/portable_bundle.safetensors"
      }
    ],
    local_report: $local_report[0],
    remote_report: $remote_report[0],
    comparison: {
      steps_per_second_gain_pct_local_over_remote:
        pct_gain($local_report[0].retained_run.steps_per_second; $remote_report[0].retained_run.steps_per_second),
      samples_per_second_gain_pct_local_over_remote:
        pct_gain($local_report[0].retained_run.samples_per_second; $remote_report[0].retained_run.samples_per_second),
      source_tokens_per_second_gain_pct_local_over_remote:
        pct_gain($local_report[0].retained_run.source_tokens_per_second; $remote_report[0].retained_run.source_tokens_per_second),
      local_to_remote_steps_ratio:
        ratio($local_report[0].retained_run.steps_per_second; $remote_report[0].retained_run.steps_per_second),
      local_to_remote_loss_delta_gap:
        ($local_report[0].retained_run.loss_delta - $remote_report[0].retained_run.loss_delta)
    },
    claim_boundary:
      "This matrix report compares the admitted home-Tailnet operator set only: the local M5 MLX path and the remote archlinux RTX 4080 CUDA path. It does not claim that sleeping or unreachable nodes were part of the retained comparison."
  }' >"${matrix_report_path}"

echo "wrote admitted-device matrix report ${matrix_report_path}"
