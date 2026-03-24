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
measurements_path=""
output_path=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-build-8xh100-receipt.sh --run-root <path> [--measurements <path>] [--output <path>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --measurements)
      measurements_path="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
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

if [[ -z "${measurements_path}" ]]; then
  measurements_path="${run_root}/parameter_golf_distributed_8xh100_measurements.json"
fi
if [[ -z "${output_path}" ]]; then
  output_path="${run_root}/parameter_golf_distributed_8xh100_receipt.json"
fi

inventory_path="${run_root}/nvidia_smi_inventory.txt"
if [[ ! -f "${inventory_path}" ]]; then
  echo "error: missing RunPod inventory at ${inventory_path}" >&2
  exit 1
fi
if [[ ! -f "${measurements_path}" ]]; then
  echo "error: missing RunPod measurements at ${measurements_path}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${output_path}")"

cargo run -q -p psionic-train --example parameter_golf_runpod_8xh100_receipt \
  --manifest-path "${repo_root}/crates/psionic-train/Cargo.toml" -- \
  "${run_root}" "${measurements_path}" "${output_path}"
