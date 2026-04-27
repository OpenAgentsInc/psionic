#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/psion/a1_minimal_distributed_lm/support_artifact_catalog_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/a1_minimal_distributed_lm_support_artifacts.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/support_artifact_catalog_v1.json"
cargo run -q -p psionic-train --example a1_minimal_distributed_lm_support_artifact_catalog_fixture -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

committed = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

if committed != generated:
    fail("A1 minimal distributed LM support artifact catalog drifted from generator output")

if committed["lane_id"] != "a1_minimal_distributed_lm_001":
    fail("support artifact catalog lost the lane id")
if committed["participant_counter_source"] != "training_accepted_contributors":
    fail("support artifact participant counter source drifted")
if committed["model_progress_counter_source"] != "training_model_progress_contributors":
    fail("support artifact model-progress counter source drifted")

expected_kinds = {
    "tokenized_shard_validation",
    "validation_replay",
    "checkpoint_verification",
    "eval_batch",
    "artifact_rematerialization",
    "independent_scored_training_window",
}
families = committed["support_families"]
family_kinds = {family["support_artifact_kind"] for family in families}
if family_kinds != expected_kinds:
    fail(f"support artifact family set drifted: {sorted(family_kinds)}")

for family in families:
    if not family["participant_eligible_on_acceptance"]:
        fail(f"{family['support_artifact_kind']} stopped being participant eligible")
    if family["model_progress_participant_by_default"]:
        fail(f"{family['support_artifact_kind']} must not be model-progress by default")
    if family["closeout_counter_source"] != "training_accepted_contributors":
        fail(f"{family['support_artifact_kind']} closeout counter source drifted")
    required_checks = set(family["validator_acceptance_checks"])
    for check in {
        "assignment_binding_matches",
        "artifact_digest_matches_payload",
        "validator_verdict_explicit",
        "closeout_verdict_explicit",
        "model_progress_counter_not_incremented",
    }:
        if check not in required_checks:
            fail(f"{family['support_artifact_kind']} lost validator check {check}")

receipts = committed["retained_example_receipts"]
receipt_kinds = {receipt["support_artifact_kind"] for receipt in receipts}
if receipt_kinds != expected_kinds:
    fail(f"support artifact receipt set drifted: {sorted(receipt_kinds)}")

for receipt in receipts:
    if receipt["validator_disposition"] != "accepted":
        fail(f"{receipt['receipt_id']} is not an accepted retained example")
    if not receipt["participant_eligible"] or not receipt["accepted_participant_work"]:
        fail(f"{receipt['receipt_id']} does not count as accepted participant work")
    if receipt["model_progress_participant"]:
        fail(f"{receipt['receipt_id']} incorrectly counts as model-progress participant work")
    if receipt["participant_counter_source"] != "training_accepted_contributors":
        fail(f"{receipt['receipt_id']} participant counter source drifted")
    if receipt["model_progress_counter_source"] != "training_model_progress_contributors":
        fail(f"{receipt['receipt_id']} model-progress counter source drifted")
    if not receipt["input_refs"] or not receipt["output_refs"]:
        fail(f"{receipt['receipt_id']} must carry explicit input and output refs")

summary = {
    "verdict": "verified",
    "lane_id": committed["lane_id"],
    "support_family_count": len(families),
    "retained_example_receipt_count": len(receipts),
    "participant_counter_source": committed["participant_counter_source"],
    "model_progress_counter_source": committed["model_progress_counter_source"],
    "catalog_digest": committed["catalog_digest"],
}
print(json.dumps(summary, indent=2))
PY
