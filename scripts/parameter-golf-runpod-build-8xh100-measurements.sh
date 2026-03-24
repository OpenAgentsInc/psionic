#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  if [[ -d "${HOME}/.cargo/bin" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required but was not found in PATH or \${HOME}/.cargo/bin" >&2
  exit 1
fi

run_root=""
log_path=""
output_path=""
memory_source="runpod execution log peak memory"

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-build-8xh100-measurements.sh --run-root <path> [--log <path>] [--output <path>] [--memory-source <text>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --log)
      log_path="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --memory-source)
      memory_source="$2"
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

if [[ -z "${run_root}" ]]; then
  echo "error: --run-root is required" >&2
  usage
  exit 1
fi

if [[ -z "${log_path}" ]]; then
  log_path="${run_root}/execution.log"
fi
if [[ -z "${output_path}" ]]; then
  output_path="${run_root}/parameter_golf_distributed_8xh100_measurements.json"
fi

if [[ ! -f "${log_path}" ]]; then
  echo "error: execution log does not exist: ${log_path}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${output_path}")"

cargo run -q -p psionic-train --example parameter_golf_runpod_8xh100_measurements_from_log \
  --manifest-path "${repo_root}/crates/psionic-train/Cargo.toml" -- \
  "${log_path}" "${output_path}" \
  --run-id "$(basename -- "${run_root}")" \
  --mesh-id "mesh.parameter_golf.runpod_8xh100" \
  --memory-source "${memory_source}"
