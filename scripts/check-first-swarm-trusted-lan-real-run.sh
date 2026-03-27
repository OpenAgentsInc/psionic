#!/usr/bin/env bash
set -euo pipefail

bundle_path=""

usage() {
  cat <<'EOF' >&2
Usage: scripts/check-first-swarm-trusted-lan-real-run.sh --bundle <path>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      bundle_path="$2"
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

if [[ -z "${bundle_path}" ]]; then
  usage
  exit 1
fi

python3 - "${bundle_path}" <<'PY'
import json
import sys
from pathlib import Path

bundle_path = Path(sys.argv[1])
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if bundle["schema_version"] != "swarm.first_trusted_lan_real_run_bundle.v1":
    fail("first swarm real-run check: schema version drifted")
if bundle["coordinator_backend_label"] != "open_adapter_backend.mlx.metal.gpt_oss_lm_head":
    fail("first swarm real-run check: coordinator backend label drifted")
if bundle["contributor_backend_label"] != "open_adapter_backend.cuda.gpt_oss_lm_head":
    fail("first swarm real-run check: contributor backend label drifted")
if bundle["submission_receipt_count"] != 2:
    fail("first swarm real-run check: expected exactly two submission receipts")
if int(bundle["accepted_contributions"]) != 2:
    fail("first swarm real-run check: expected two accepted contributions")
if int(bundle["replay_checked_contributions"]) != 2:
    fail("first swarm real-run check: expected replay_checked_contributions=2")
if bundle["merge_disposition"] != "merged":
    fail("first swarm real-run check: merge_disposition must stay merged for the retained run")
if bundle["publish_disposition"] != "refused":
    fail("first swarm real-run check: publish_disposition must stay refused for the retained run")
if len(bundle.get("replay_receipt_digests", [])) != 2:
    fail("first swarm real-run check: expected exactly two replay receipt digests")
if "mixed-hardware open-adapter run" not in bundle["claim_boundary"]:
    fail("first swarm real-run check: claim boundary drifted")

summary = {
    "verdict": "verified",
    "run_id": bundle["run_id"],
    "result_classification": bundle["result_classification"],
    "promotion_disposition": bundle["promotion_disposition"],
    "merge_disposition": bundle["merge_disposition"],
    "publish_disposition": bundle["publish_disposition"],
}
print(json.dumps(summary, indent=2))
PY
