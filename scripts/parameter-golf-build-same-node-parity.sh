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

psionic_report=""
upstream_log=""
output_dir=""
upstream_run_id="train-gpt-reference"
device_name=""
world_size="1"
train_batch_tokens="524288"
validation_batch_tokens="524288"
train_sequence_length="1024"
grad_accum_steps="8"

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-build-same-node-parity.sh --psionic-report <path> --upstream-log <path> --output-dir <path> [options]

Options:
  --upstream-run-id <id>            Stable upstream run id. Default: train-gpt-reference
  --device-name <name>              Override the upstream device name. Default: first device name from the Psionic report.
  --world-size <n>                  Batch geometry world size. Default: 1
  --train-batch-tokens <n>          Batch geometry train batch tokens. Default: 524288
  --validation-batch-tokens <n>     Batch geometry validation batch tokens. Default: 524288
  --train-sequence-length <n>       Batch geometry train sequence length. Default: 1024
  --grad-accum-steps <n>            Batch geometry grad accumulation steps. Default: 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --psionic-report)
      psionic_report="$2"
      shift 2
      ;;
    --upstream-log)
      upstream_log="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --upstream-run-id)
      upstream_run_id="$2"
      shift 2
      ;;
    --device-name)
      device_name="$2"
      shift 2
      ;;
    --world-size)
      world_size="$2"
      shift 2
      ;;
    --train-batch-tokens)
      train_batch_tokens="$2"
      shift 2
      ;;
    --validation-batch-tokens)
      validation_batch_tokens="$2"
      shift 2
      ;;
    --train-sequence-length)
      train_sequence_length="$2"
      shift 2
      ;;
    --grad-accum-steps)
      grad_accum_steps="$2"
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

if [[ -z "${psionic_report}" || -z "${upstream_log}" || -z "${output_dir}" ]]; then
  echo "error: --psionic-report, --upstream-log, and --output-dir are required" >&2
  usage
  exit 1
fi

if [[ ! -f "${psionic_report}" ]]; then
  echo "error: Psionic report does not exist: ${psionic_report}" >&2
  exit 1
fi
if [[ ! -f "${upstream_log}" ]]; then
  echo "error: upstream log does not exist: ${upstream_log}" >&2
  exit 1
fi

mkdir -p "${output_dir}"

readarray -t psionic_metadata < <(
  python3 - "${psionic_report}" "${device_name}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
device_override = sys.argv[2]
report = json.loads(report_path.read_text(encoding="utf-8"))
device_name = device_override
if not device_name:
    for device in report.get("observed_cuda_devices", []):
        if device.get("device_name"):
            device_name = device["device_name"]
            break
if not device_name:
    raise SystemExit("missing device name in both --device-name and Psionic report")
print(report["dataset_manifest_digest"])
print(report["tokenizer_digest"]["tokenizer_digest"])
print(device_name)
PY
)

dataset_manifest_digest="${psionic_metadata[0]}"
tokenizer_digest="${psionic_metadata[1]}"
resolved_device_name="${psionic_metadata[2]}"

upstream_receipt="${output_dir}/parameter_golf_train_gpt_reference_run_receipt.json"
parity_report="${output_dir}/parameter_golf_same_node_parity_report.json"

cargo run -q -p psionic-train --example parameter_golf_train_gpt_reference_run_receipt -- \
  --run-id "${upstream_run_id}" \
  --log "${upstream_log}" \
  --output "${upstream_receipt}" \
  --device-name "${resolved_device_name}" \
  --dataset-manifest-digest "${dataset_manifest_digest}" \
  --tokenizer-digest "${tokenizer_digest}" \
  --world-size "${world_size}" \
  --train-batch-tokens "${train_batch_tokens}" \
  --validation-batch-tokens "${validation_batch_tokens}" \
  --train-sequence-length "${train_sequence_length}" \
  --grad-accum-steps "${grad_accum_steps}"

cargo run -q -p psionic-train --example parameter_golf_same_node_parity_report -- \
  "${psionic_report}" \
  "${upstream_receipt}" \
  "${parity_report}"

printf 'upstream_receipt=%s\nparity_report=%s\n' "${upstream_receipt}" "${parity_report}"
