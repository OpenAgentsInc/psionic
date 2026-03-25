#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_path=""

usage() {
  cat <<'EOF' >&2
Usage: check-first-swarm-trusted-lan-evidence-bundle.sh [--report <path>]
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
  report_path="$(mktemp "${TMPDIR:-/tmp}/first_swarm_trusted_lan_evidence_bundle.XXXXXX")"
  cleanup_report=1
fi

cleanup() {
  if [[ "${cleanup_report}" -eq 1 ]]; then
    rm -f -- "${report_path}"
  fi
}
trap cleanup EXIT

cargo run -q -p psionic-train --bin first_swarm_trusted_lan_evidence_bundle -- "${report_path}"

python3 - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

bundle_path = Path(sys.argv[1])
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if bundle["schema_version"] != "swarm.first_trusted_lan_evidence_bundle.v1":
    fail("first swarm evidence-bundle check: schema version drifted")
if bundle["live_attempt_disposition"] != "refused":
    fail("first swarm evidence-bundle check: live attempt must stay refused for the current lane")
if bundle["launch_status"] != "bundle_materialized":
    fail("first swarm evidence-bundle check: launch status drifted")
if bundle["promotion_outcome"]["disposition"] != "no_promotion":
    fail("first swarm evidence-bundle check: promotion outcome must stay no_promotion")
if bundle["promotion_outcome"].get("local_snapshot_path") is not None:
    fail("first swarm evidence-bundle check: refused bundle must not carry a local snapshot path")
if "refused first live attempt" not in bundle["claim_boundary"]:
    fail("first swarm evidence-bundle check: claim boundary drifted")
contributors = bundle["contributors"]
if len(contributors) != 2:
    fail("first swarm evidence-bundle check: expected exactly two contributor rows")
for contributor in contributors:
    if contributor["upload_posture"] != "planned_only":
        fail("first swarm evidence-bundle check: contributors must stay planned_only in the refused bundle")
    if contributor["validator_posture"] != "not_executed":
        fail("first swarm evidence-bundle check: validator posture must stay not_executed")
    if contributor["aggregation_posture"] != "not_executed":
        fail("first swarm evidence-bundle check: aggregation posture must stay not_executed")
    if contributor["replay_posture"] != "not_executed":
        fail("first swarm evidence-bundle check: replay posture must stay not_executed")
stages = bundle["stages"]
stage_ids = {stage["stage_id"] for stage in stages}
required_stage_ids = {
    "operator_bundle_materialization",
    "live_attempt_gate",
    "contributor_execution",
    "upload_validation_aggregation",
    "promotion_closeout",
}
if stage_ids != required_stage_ids:
    fail("first swarm evidence-bundle check: stage set drifted")
gate_stage = next(stage for stage in stages if stage["stage_id"] == "live_attempt_gate")
if gate_stage["disposition"] != "refused":
    fail("first swarm evidence-bundle check: live_attempt_gate must stay refused")
if bundle["rehearsal_report_digest"] == "":
    fail("first swarm evidence-bundle check: rehearsal report digest missing")
if bundle["failure_drills_digest"] == "":
    fail("first swarm evidence-bundle check: failure-drill digest missing")
if bundle["workflow_plan_digest"] == "":
    fail("first swarm evidence-bundle check: workflow-plan digest missing")

summary = {
    "verdict": "verified",
    "bundle_digest": bundle["bundle_digest"],
    "live_attempt_disposition": bundle["live_attempt_disposition"],
    "promotion_outcome": bundle["promotion_outcome"]["disposition"],
    "stage_count": len(stages),
    "topology_contract_digest": bundle["topology_contract_digest"],
}
print(json.dumps(summary, indent=2))
PY
