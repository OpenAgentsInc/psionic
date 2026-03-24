#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_path=""

usage() {
  cat <<'EOF' >&2
Usage: check-parameter-golf-runpod-8xh100-lane.sh [--report <path>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      report_path="$2"
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
  report_path="$(mktemp "${TMPDIR:-/tmp}/parameter_golf_runpod_8xh100_rehearsal.XXXXXX.json")"
  cleanup_report=1
else
  cleanup_report=0
fi

cleanup() {
  if [[ "${cleanup_report}" -eq 1 ]]; then
    rm -f -- "${report_path}"
  fi
}
trap cleanup EXIT

preflight_json="$(
  bash "${repo_root}/scripts/parameter-golf-runpod-operator-preflight.sh" \
    --profile runpod_8xh100_parameter_golf
)"

manifest_json="$(
  bash "${repo_root}/scripts/parameter-golf-runpod-launch-8xh100.sh" \
    --profile runpod_8xh100_parameter_golf \
    --run-id parameter-golf-runpod-rehearsal \
    --manifest-only
)"

python3 - "${repo_root}" "${report_path}" "${preflight_json}" "${manifest_json}" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
report_path = Path(sys.argv[2])
preflight = json.loads(sys.argv[3])
manifest = json.loads(sys.argv[4])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if preflight.get("result") != "ready":
    fail("RunPod preflight did not report ready")
if manifest.get("profile_id") != "runpod_8xh100_parameter_golf":
    fail("launch manifest drifted to the wrong profile")
if manifest.get("trainer_lane_id") != "parameter_golf_distributed_8xh100":
    fail("launch manifest drifted to the wrong trainer lane")
if manifest.get("expected_execution_backend") != "cuda":
    fail("launch manifest lost the CUDA execution claim")

topology = manifest.get("topology") or {}
if topology.get("accelerator_count") != 8:
    fail("launch manifest lost the 8-device requirement")
if topology.get("world_size") != 8:
    fail("launch manifest lost WORLD_SIZE=8")
if topology.get("grad_accum_steps") != 1:
    fail("launch manifest lost grad_accum_steps=1")

commands = manifest.get("commands") or {}
if "python3 train_gpt.py" not in (commands.get("execution_entrypoint_command") or ""):
    fail("launch manifest execution entrypoint no longer uses the exported folder surface")
if "parameter-golf-runpod-finalize-8xh100.sh" not in (commands.get("finalizer_command") or ""):
    fail("launch manifest finalizer contract drifted")

receipts = manifest.get("expected_receipt_paths") or []
if not any("nvidia_smi_inventory.txt" in path for path in receipts):
    fail("launch manifest no longer preserves GPU inventory evidence")
if not any("psionic_parameter_golf_submission_run_evidence.json" in path for path in receipts):
    fail("launch manifest no longer preserves exported-folder run evidence")

report = {
    "schema_version": "parameter_golf.runpod_8xh100_operator_rehearsal.v1",
    "runner": "scripts/check-parameter-golf-runpod-8xh100-lane.sh",
    "profile_id": manifest["profile_id"],
    "trainer_lane_id": manifest["trainer_lane_id"],
    "preflight_result": preflight["result"],
    "accelerator_type": topology["accelerator_type"],
    "accelerator_count": topology["accelerator_count"],
    "world_size": topology["world_size"],
    "grad_accum_steps": topology["grad_accum_steps"],
    "input_package_descriptor_uri": manifest["input_package"]["descriptor_uri"],
    "execution_entrypoint_command": commands["execution_entrypoint_command"],
    "finalizer_command": commands["finalizer_command"],
    "manifest_only_launch_supported": manifest["manifest_only"],
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
