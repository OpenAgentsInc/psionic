#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: psion-actual-pretraining-status.sh --run-root <path>

Reads the canonical actual-lane status surfaces from:
  <run-root>/status/current_run_status.json
  <run-root>/status/retained_summary.json
EOF
}

run_root=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="$2"
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

status_path="${run_root}/status/current_run_status.json"
summary_path="${run_root}/status/retained_summary.json"

if [[ ! -f "${status_path}" ]]; then
  echo "error: missing current status surface at ${status_path}" >&2
  exit 1
fi
if [[ ! -f "${summary_path}" ]]; then
  echo "error: missing retained summary surface at ${summary_path}" >&2
  exit 1
fi

python3 - <<'PY' "${status_path}" "${summary_path}"
import json
import sys

status_path, summary_path = sys.argv[1:]

with open(status_path, "r", encoding="utf-8") as handle:
    status = json.load(handle)
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)

print(f"run_id={status.get('run_id')}")
print(f"phase={status.get('phase')}")
print(f"last_completed_step={status.get('last_completed_step')}")
print(f"latest_checkpoint_label={status.get('latest_checkpoint_label')}")
print(f"selected_git_ref={summary.get('selected_git_ref')}")
print(f"git_commit_sha={summary.get('git_commit_sha')}")
print(f"dirty_tree_admission={summary.get('dirty_tree_admission')}")
print(f"status_surface_id={summary.get('launcher_surfaces', {}).get('status_surface_id')}")
PY
