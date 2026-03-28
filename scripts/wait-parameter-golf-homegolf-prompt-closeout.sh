#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

report_path=""
prompt="the meaning of life is"
max_new_tokens="32"
poll_seconds="30"
timeout_seconds="0"
output_path=""
summary_path=""
binary_path=""

usage() {
  cat <<'EOF' >&2
Usage: scripts/wait-parameter-golf-homegolf-prompt-closeout.sh --report <path> [options]

Options:
  --report <path>                      Training report to wait for.
  --prompt <text>                      Prompt for post-run generation proof.
  --max-new-tokens <n>                 Max generated tokens. Default: 32
  --poll-seconds <n>                   Poll interval while waiting. Default: 30
  --timeout-seconds <n>                Optional timeout. Default: 0 (no timeout)
  --output <path>                      Prompt report output path.
  --summary <path>                     Combined summary output path.
  --binary-path <path>                 Optional prompt binary path.
  --help|-h                            Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      report_path="$2"
      shift 2
      ;;
    --prompt)
      prompt="$2"
      shift 2
      ;;
    --max-new-tokens)
      max_new_tokens="$2"
      shift 2
      ;;
    --poll-seconds)
      poll_seconds="$2"
      shift 2
      ;;
    --timeout-seconds)
      timeout_seconds="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --summary)
      summary_path="$2"
      shift 2
      ;;
    --binary-path)
      binary_path="$2"
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

if [[ -z "${report_path}" ]]; then
  echo "error: --report is required" >&2
  usage
  exit 1
fi

report_path="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${report_path}")"
report_dir="$(dirname "${report_path}")"
report_stem="$(basename "${report_path}" .json)"

if [[ -z "${output_path}" ]]; then
  output_path="${report_dir}/${report_stem}_prompt.json"
fi
if [[ -z "${summary_path}" ]]; then
  summary_path="${report_dir}/${report_stem}_closeout_summary.json"
fi

mkdir -p "$(dirname "${output_path}")" "$(dirname "${summary_path}")"

start_epoch="$(date +%s)"
while true; do
  if [[ -f "${report_path}" ]]; then
    if python3 - <<'PY' "${report_path}" >/dev/null 2>&1
import json, sys
json.load(open(sys.argv[1], "r", encoding="utf-8"))
PY
    then
      break
    fi
  fi
  if [[ "${timeout_seconds}" != "0" ]]; then
    now_epoch="$(date +%s)"
    elapsed="$((now_epoch - start_epoch))"
    if (( elapsed >= timeout_seconds )); then
      echo "error: timed out waiting for ${report_path}" >&2
      exit 1
    fi
  fi
  sleep "${poll_seconds}"
done

prompt_command=()
if [[ -n "${binary_path}" ]]; then
  if [[ ! -x "${binary_path}" ]]; then
    echo "error: binary path ${binary_path} is not executable" >&2
    exit 1
  fi
  prompt_command=("${binary_path}")
else
  prompt_command=(
    cargo
    run
    -q
    -p
    psionic-train
    --bin
    parameter_golf_homegolf_prompt
    --
  )
fi

cd "${repo_root}"
"${prompt_command[@]}" \
  "${report_path}" \
  "${prompt}" \
  "${max_new_tokens}" \
  "${output_path}" >/dev/null

python3 - <<'PY' "${report_path}" "${output_path}" "${summary_path}"
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
prompt_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

report = json.loads(report_path.read_text(encoding="utf-8"))
prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
final_roundtrip = report.get("final_roundtrip_receipt") or {}
validation = final_roundtrip.get("validation") or {}
summary = {
    "schema_version": "psionic.homegolf_prompt_closeout_summary.v1",
    "report_path": str(report_path),
    "prompt_report_path": str(prompt_path),
    "run_id": report.get("run_id"),
    "machine_profile": report.get("machine_profile"),
    "model_variant": report.get("model_variant"),
    "disposition": report.get("disposition"),
    "executed_steps": report.get("executed_steps"),
    "observed_training_time_ms": report.get("observed_training_time_ms"),
    "compressed_model_artifact_path": report.get("compressed_model_artifact_path"),
    "compressed_model_bytes": report.get("compressed_model_bytes"),
    "final_validation_bits_per_byte": validation.get("bits_per_byte"),
    "final_validation_mean_loss": validation.get("mean_loss"),
    "final_roundtrip_eval_ms": final_roundtrip.get("observed_eval_ms"),
    "prompt": prompt.get("prompt"),
    "generated_text": prompt.get("generated_text"),
    "termination": prompt.get("termination"),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
