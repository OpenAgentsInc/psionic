#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/collective_failure_semantics_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/collective_failure_semantics.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/collective_failure_semantics_v1.json"
cargo run -q -p psionic-collectives --bin collective_failure_semantics_contract -- "${generated_path}" >/dev/null

python3 - "${contract_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

committed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generated = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if committed != generated:
    fail("collective failure semantics check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.collectives.failure_semantics.v1":
    fail("collective failure semantics check: schema_version drifted")
if committed["contract_id"] != "psionic.collectives.failure_semantics.v1":
    fail("collective failure semantics check: contract_id drifted")
if committed["issue_ref"] != "psionic#1126":
    fail("collective failure semantics check: issue_ref drifted")

plan = committed["chunk_plan"]
if plan["chunk_count"] != 8 or plan["last_chunk_bytes"] != 2 * 1024 * 1024:
    fail("collective failure semantics check: chunk plan drifted")
if plan["per_chunk_timeout_ms"] != committed["config"]["per_chunk_timeout_ms"]:
    fail("collective failure semantics check: send-path chunk budget drifted from config")
if plan["reciprocal_per_chunk_timeout_ms"] != committed["config"]["reciprocal_per_chunk_timeout_ms"]:
    fail("collective failure semantics check: reciprocal-path chunk budget drifted from config")

receipts = committed["round_receipts"]
if len(receipts) != 7:
    fail("collective failure semantics check: expected exactly seven round receipts")
verdicts = [receipt["verdict"]["verdict"] for receipt in receipts]
if verdicts != ["complete", "partial", "partial", "partial", "aborted", "partial", "partial"]:
    fail("collective failure semantics check: round verdict sequence drifted")

# Ban-for-round, not retry: the send-path failure round preserves prior chunks.
send_failure = receipts[1]
if send_failure["verdict"]["banned_node_ids"] != ["worker-b"]:
    fail("collective failure semantics check: send-failure ban drifted")
if send_failure["verdict"]["preserved_fraction_bps"] != 7500:
    fail("collective failure semantics check: send-failure preserved fraction drifted")
worker_b = next(
    contribution for contribution in send_failure["included_contributions"]
    if contribution["member_node_id"] == "worker-b"
)
if worker_b["delivered_chunk_indices"] != [0, 1] or not worker_b["banned_after_contribution"]:
    fail("collective failure semantics check: pre-ban chunk preservation drifted")

# Reciprocal return-path timeout uses the same ban semantics.
reciprocal_failure = receipts[2]
if reciprocal_failure["ban_events"][0]["reason"] != "reciprocal_chunk_timed_out":
    fail("collective failure semantics check: reciprocal ban reason drifted")
if reciprocal_failure["ban_events"][0]["path"] != "reciprocal_return":
    fail("collective failure semantics check: reciprocal ban path drifted")
if reciprocal_failure["verdict"]["preserved_fraction_bps"] != 9166:
    fail("collective failure semantics check: reciprocal preserved fraction drifted")

# Stage-empty without a promotable standby aborts.
aborted = receipts[4]
if aborted["verdict"]["reason"] != "no_standby_promotable":
    fail("collective failure semantics check: abort reason drifted")
if aborted["standby_decision"]["promotable"]:
    fail("collective failure semantics check: abort standby decision drifted")

# Stage-empty with a warm standby stays partial and records the promotion.
promoted = receipts[5]
if promoted["verdict"]["promoted_standby_node_id"] != "standby-w1":
    fail("collective failure semantics check: standby promotion drifted")
if not promoted["standby_decision"]["promotable"]:
    fail("collective failure semantics check: promotable standby decision drifted")

# Full-round timeout retains the partial result.
timed_out = receipts[6]
if not timed_out["round_timed_out"]:
    fail("collective failure semantics check: round-timeout flag drifted")
if timed_out["verdict"]["preserved_fraction_bps"] != 8333:
    fail("collective failure semantics check: round-timeout preserved fraction drifted")
if timed_out["ban_events"][0]["reason"] != "full_round_timeout_stall":
    fail("collective failure semantics check: round-timeout stall ban drifted")

removals = committed["removal_recommendations"]
if [removal["member_node_id"] for removal in removals] != ["worker-b", "worker-c"]:
    fail("collective failure semantics check: removal recommendations drifted")
if removals[0]["ban_count_in_window"] != committed["ban_escalation_policy"]["removal_ban_threshold"]:
    fail("collective failure semantics check: removal threshold accounting drifted")

paths = committed["authority_paths"]
if paths["check_script_path"] != "scripts/check-collective-failure-semantics.sh":
    fail("collective failure semantics check: check_script_path drifted")
if paths["reference_doc_path"] != "docs/PSIONIC_COLLECTIVE_FAILURE_SEMANTICS.md":
    fail("collective failure semantics check: reference_doc_path drifted")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "round_verdicts": verdicts,
    "removal_recommendations": [removal["member_node_id"] for removal in removals],
}
print(json.dumps(summary, indent=2))
PY
