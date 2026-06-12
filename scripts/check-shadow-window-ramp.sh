#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/shadow_window_ramp_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/shadow_window_ramp.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/shadow_window_ramp_v1.json"
cargo run -q -p psionic-train --bin shadow_window_ramp_contract -- "${generated_path}" >/dev/null

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
    fail("shadow window ramp check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.train.shadow_window_ramp.v1":
    fail("shadow window ramp check: schema_version drifted")
if committed["contract_id"] != "psionic.train.shadow_window_ramp.v1":
    fail("shadow window ramp check: contract_id drifted")

config = committed["ramp_config"]
if config["phase1_shadow_replay_window_count"] < 1 or config["phase2_live_shadow_window_count"] < 1:
    fail("shadow window ramp check: ramp phase lengths must be at least 1")

binding = committed["checkpoint_binding"]
for field in ("checkpoint_object_digest", "manifest_digest"):
    digest = binding[field]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        fail(f"shadow window ramp check: {field} is not a SHA256 hex digest")

receipts = committed["shadow_receipts"]
shadow = [r for r in receipts if r["phase"] in ("shadow_replay", "live_shadow")]
active = [r for r in receipts if r["phase"] == "active"]
if not shadow or not active:
    fail("shadow window ramp check: expected both shadow-phase and active-phase receipts")
for receipt in shadow:
    if receipt["merge_eligibility"] != "shadow":
        fail(f"shadow window ramp check: shadow-phase receipt `{receipt['receipt_id']}` claims merge eligibility")
for receipt in active:
    if receipt["merge_eligibility"] != "merge_eligible":
        fail(f"shadow window ramp check: active-phase receipt `{receipt['receipt_id']}` is not merge eligible")

shadow_ids = {r["receipt_id"] for r in shadow}
merged_ids = set(committed["merge_set_member_receipt_ids"])
excluded_ids = set(committed["merge_excluded_receipt_ids"])
if merged_ids & shadow_ids:
    fail("shadow window ramp check: a shadow receipt leaked into the merge set")
if excluded_ids != shadow_ids:
    fail("shadow window ramp check: merge exclusions must be exactly the shadow receipts")
if merged_ids != {r["receipt_id"] for r in active}:
    fail("shadow window ramp check: merge set must be exactly the active receipts")
if committed["final_ramp_state"]["phase"] != "active":
    fail("shadow window ramp check: replayed ramp must finish in the active phase")

comparison = committed["divergence_comparison"]
labels = sorted(cohort["cohort_label"] for cohort in comparison["cohorts"])
if labels != ["bootstrap_and_merge", "ramped_join"]:
    fail("shadow window ramp check: expected one ramped-join and one bootstrap-and-merge cohort")
if comparison["measurement_status"] != "synthetic_fixture":
    fail("shadow window ramp check: no measured divergence run exists; status must stay synthetic_fixture")
if "synthetic" not in comparison["detail"].lower():
    fail("shadow window ramp check: synthetic comparison must say it is synthetic")
if "synthetic" not in committed["claim_boundary"].lower():
    fail("shadow window ramp check: claim_boundary must keep the synthetic boundary explicit")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "ramp_config": config,
    "merge_set_member_receipt_ids": sorted(merged_ids),
    "merge_excluded_receipt_count": len(excluded_ids),
    "measurement_status": comparison["measurement_status"],
}
print(json.dumps(summary, indent=2))
PY
