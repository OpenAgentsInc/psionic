#!/usr/bin/env bash

set -euo pipefail

BUCKET_URL="${BUCKET_URL:-gs://openagentsgemini-psion-train-us-central1}"
BUNDLE_PATH="${BUNDLE_PATH:-}"
RUN_ID="${RUN_ID:-}"

usage() {
  cat <<'EOF'
Usage: check-psion-google-two-node-swarm-evidence-bundle.sh [options]

Options:
  --bundle <path|gs://uri>    Validate one explicit evidence bundle.
  --run-id <run_id>           Resolve the evidence bundle from the training bucket.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      BUNDLE_PATH="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${BUNDLE_PATH}" && -z "${RUN_ID}" ]]; then
  echo "error: provide --bundle or --run-id" >&2
  exit 1
fi

if [[ -z "${BUNDLE_PATH}" ]]; then
  BUNDLE_PATH="${BUCKET_URL}/runs/${RUN_ID}/final/psion_google_two_node_swarm_evidence_bundle.json"
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
local_bundle="${tmpdir}/bundle.json"

if [[ "${BUNDLE_PATH}" == gs://* ]]; then
  gcloud storage cp --quiet "${BUNDLE_PATH}" "${local_bundle}" >/dev/null
else
  cp "${BUNDLE_PATH}" "${local_bundle}"
fi

python3 - "${local_bundle}" <<'PY'
import json
import sys
from pathlib import Path

bundle_path = Path(sys.argv[1])
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if bundle["schema_version"] != "psion.google_two_node_swarm_evidence_bundle.v1":
    fail("google two-node swarm evidence check: schema version drifted")
if bundle["result_classification"] not in {
    "configured_peer_launch_failure",
    "cluster_membership_failure",
    "network_impairment_gate_failure",
    "contributor_execution_failure",
    "validator_refusal",
    "aggregation_failure",
    "bounded_success",
}:
    fail("google two-node swarm evidence check: unexpected result classification")
if len(bundle["topology"]["nodes"]) != 2:
    fail("google two-node swarm evidence check: expected exactly two topology nodes")
if len(bundle["bringup_reports"]) != 2:
    fail("google two-node swarm evidence check: expected exactly two bring-up reports")
if len(bundle["runtime_reports"]) != 2:
    fail("google two-node swarm evidence check: expected exactly two runtime reports")

runtime_roles = {report["runtime_role"] for report in bundle["runtime_reports"]}
if runtime_roles != {"coordinator", "contributor"}:
    fail("google two-node swarm evidence check: runtime roles drifted")

bringup_statuses = {report["status"] for report in bundle["bringup_reports"]}
if not bringup_statuses:
    fail("google two-node swarm evidence check: bring-up statuses missing")

retained_kinds = {record["artifact_kind"] for record in bundle["retained_objects"]}
required_kinds = {
    "cluster_manifest",
    "launch_receipt",
    "coordinator_bringup_report",
    "contributor_bringup_report",
    "coordinator_runtime_report",
    "contributor_runtime_report",
}
missing = required_kinds - retained_kinds
if missing:
    fail(
        "google two-node swarm evidence check: retained objects missing "
        + ", ".join(sorted(missing))
    )

profile_id = bundle["selected_impairment_profile_id"]
if profile_id != "clean_baseline" and len(bundle["impairment_receipts"]) != 2:
    fail(
        "google two-node swarm evidence check: non-clean impairment profile requires two retained impairment receipts"
    )

cluster_health = bundle["cluster_health"]
if bundle["result_classification"] == "bounded_success":
    if bundle["validator_posture"] is None:
        fail("google two-node swarm evidence check: bounded success requires validator posture")
    if bundle["aggregation_posture"] is None:
        fail("google two-node swarm evidence check: bounded success requires aggregation posture")
    if cluster_health["submission_receipt_count"] < 2:
        fail("google two-node swarm evidence check: bounded success requires at least two submission receipts")

summary = {
    "verdict": "verified",
    "run_id": bundle["run_id"],
    "cluster_id": bundle["cluster_id"],
    "result_classification": bundle["result_classification"],
    "selected_impairment_profile_id": profile_id,
    "submission_receipt_count": cluster_health["submission_receipt_count"],
    "retained_object_count": len(bundle["retained_objects"]),
}
print(json.dumps(summary, indent=2))
PY
