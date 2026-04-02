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
import os
import sys

status_path, summary_path = sys.argv[1:]
run_root = os.path.dirname(os.path.dirname(status_path))

with open(status_path, "r", encoding="utf-8") as handle:
    status = json.load(handle)
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)

latest_eval_path = os.path.join(run_root, "evals", "latest_checkpoint_eval_decision.json")
latest_failure_path = os.path.join(run_root, "evals", "latest_checkpoint_eval_failure.json")
latest_alert_path = os.path.join(run_root, "alerts", "latest_redacted_alert.json")
latest_decision_path = os.path.join(run_root, "decisions", "latest_continue_restart_decision.json")

latest_eval = None
latest_failure = None
latest_alert = None
latest_decision = None
if os.path.exists(latest_eval_path):
    with open(latest_eval_path, "r", encoding="utf-8") as handle:
        latest_eval = json.load(handle)
if os.path.exists(latest_failure_path):
    with open(latest_failure_path, "r", encoding="utf-8") as handle:
        latest_failure = json.load(handle)
if os.path.exists(latest_alert_path):
    with open(latest_alert_path, "r", encoding="utf-8") as handle:
        latest_alert = json.load(handle)
if os.path.exists(latest_decision_path):
    with open(latest_decision_path, "r", encoding="utf-8") as handle:
        latest_decision = json.load(handle)

print(f"run_id={status.get('run_id')}")
print(f"phase={status.get('phase')}")
print(f"last_completed_step={status.get('last_completed_step')}")
print(f"latest_checkpoint_label={status.get('latest_checkpoint_label')}")
print(f"selected_git_ref={summary.get('selected_git_ref')}")
print(f"git_commit_sha={summary.get('git_commit_sha')}")
print(f"dirty_tree_admission={summary.get('dirty_tree_admission')}")
print(f"status_surface_id={summary.get('launcher_surfaces', {}).get('status_surface_id')}")
if latest_eval is not None:
    print(f"checkpoint_eval_decision_state={latest_eval.get('decision_state')}")
    print(f"checkpoint_eval_score_bps={latest_eval.get('aggregate_score_bps')}")
if latest_failure is not None:
    print(f"checkpoint_eval_failure_kind={latest_failure.get('failure_kind')}")
    print(f"checkpoint_eval_resolution_state={latest_failure.get('resolution_state')}")
if latest_alert is not None:
    print(f"latest_alert_kind={latest_alert.get('alert_kind')}")
    print(f"latest_alert_severity={latest_alert.get('severity')}")
if latest_decision is not None:
    print(f"continue_restart_decision_state={latest_decision.get('decision_state')}")
    print(f"continue_restart_operator_action={latest_decision.get('operator_action')}")
    print(f"continue_restart_blocking_row_count={len(latest_decision.get('blocking_row_ids', []))}")
PY
