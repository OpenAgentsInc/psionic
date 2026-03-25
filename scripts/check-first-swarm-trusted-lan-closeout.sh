#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_path=""

usage() {
  cat <<'EOF' >&2
Usage: check-first-swarm-trusted-lan-closeout.sh [--report <path>]
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
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cleanup_report=0
if [[ -z "${report_path}" ]]; then
  report_path="$(mktemp "${TMPDIR:-/tmp}/first_swarm_trusted_lan_closeout.XXXXXX")"
  cleanup_report=1
fi

cleanup() {
  if [[ "${cleanup_report}" -eq 1 ]]; then
    rm -f -- "${report_path}"
  fi
}
trap cleanup EXIT

cargo run -q -p psionic-train --bin first_swarm_trusted_lan_closeout_report -- "${report_path}"

python3 - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if report["schema_version"] != "swarm.first_trusted_lan_closeout_report.v1":
    fail("first swarm closeout check: schema version drifted")
if report["merge_disposition"] != "no_merge":
    fail("first swarm closeout check: merge disposition must stay no_merge")
if report["publish_disposition"] != "refused":
    fail("first swarm closeout check: publish disposition must stay refused")
if report["promotion_disposition"] != "no_promotion":
    fail("first swarm closeout check: promotion disposition must stay no_promotion")
if report.get("published_snapshot_path") is not None:
    fail("first swarm closeout check: refused closeout must not publish a snapshot path")
publish_expectation = report["publish_expectation"]
if publish_expectation["publish_id"] != "first-swarm-local-snapshot":
    fail("first swarm closeout check: publish id drifted")
if publish_expectation["target"] != "hugging_face_snapshot":
    fail("first swarm closeout check: publish target drifted")
if publish_expectation["publish_surface"] != "psionic-mlx-workflows::MlxWorkflowWorkspace::publish_bundle":
    fail("first swarm closeout check: publish surface drifted")
gates = {gate["gate_id"]: gate for gate in report["closeout_gates"]}
required_gates = {
    "all_required_contributor_roles_present",
    "validator_acceptance_receipts_exist",
    "replay_receipts_exist_for_accepted_contributions",
    "aggregation_completed_under_policy",
    "promotion_earned_local_snapshot",
}
if set(gates) != required_gates:
    fail("first swarm closeout check: gate set drifted")
if any(gate["satisfied"] for gate in gates.values()):
    fail("first swarm closeout check: all closeout gates must stay unsatisfied for the current lane")
if "no-merge truth" not in report["claim_boundary"]:
    fail("first swarm closeout check: claim boundary drifted")

summary = {
    "verdict": "verified",
    "report_digest": report["report_digest"],
    "merge_disposition": report["merge_disposition"],
    "publish_disposition": report["publish_disposition"],
    "publish_directory": publish_expectation["expected_local_snapshot_directory"],
}
print(json.dumps(summary, indent=2))
PY
