#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: psion-actual-pretraining-dashboard.sh --run-root <path>

Reads the canonical actual-lane retained dashboard surfaces from:
  <run-root>/dashboard/current_dashboard.json
  <run-root>/alerts/active_alerts.json
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

dashboard_path="${run_root}/dashboard/current_dashboard.json"
alert_feed_path="${run_root}/alerts/active_alerts.json"

if [[ ! -f "${dashboard_path}" ]]; then
  echo "error: missing retained dashboard at ${dashboard_path}" >&2
  exit 1
fi
if [[ ! -f "${alert_feed_path}" ]]; then
  echo "error: missing retained active-alert feed at ${alert_feed_path}" >&2
  exit 1
fi

python3 - <<'PY' "${dashboard_path}" "${alert_feed_path}"
import json
import sys

dashboard_path, alert_feed_path = sys.argv[1:]

with open(dashboard_path, "r", encoding="utf-8") as handle:
    dashboard = json.load(handle)
with open(alert_feed_path, "r", encoding="utf-8") as handle:
    alert_feed = json.load(handle)

print(f"run_id={dashboard.get('run_id')}")
print(f"phase={dashboard.get('current_phase')}")
print(f"selected_git_ref={dashboard.get('selected_git_ref')}")
print(f"git_commit_sha={dashboard.get('git_commit_sha')}")
print(f"throughput_state={dashboard.get('throughput', {}).get('degradation_state')}")
print(f"observed_tokens_per_second={dashboard.get('throughput', {}).get('observed_tokens_per_second')}")
print(f"baseline_tokens_per_second={dashboard.get('throughput', {}).get('baseline_tokens_per_second')}")
print(f"loss_visibility_state={dashboard.get('loss', {}).get('visibility_state')}")
print(f"gradient_visibility_state={dashboard.get('gradient', {}).get('visibility_state')}")
print(f"checkpoint_label={dashboard.get('checkpoint', {}).get('checkpoint_label')}")
print(f"checkpoint_backup_state={dashboard.get('checkpoint', {}).get('checkpoint_backup_state')}")
print(f"checkpoint_eval_state={dashboard.get('checkpoint', {}).get('checkpoint_eval_state')}")
print(f"hardware_health_state={dashboard.get('hardware', {}).get('health_state')}")
print(f"active_alert_count={dashboard.get('alerts', {}).get('active_alert_count')}")
print(f"highest_alert_severity={dashboard.get('alerts', {}).get('highest_severity')}")
for index, alert in enumerate(alert_feed.get("active_alerts", [])):
    kind = alert.get("alert_kind")
    severity = alert.get("severity")
    source = alert.get("source_relative_path")
    print(f"alert[{index}]={kind}:{severity}:{source}")
PY
