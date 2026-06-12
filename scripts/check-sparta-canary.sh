#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/sparta_canary_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/sparta_canary.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/sparta_canary_v1.json"
cargo run -q -p psionic-train --bin sparta_canary_contract -- "${generated_path}" >/dev/null

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
    fail("sparta canary check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.train.sparta_canary.v1":
    fail("sparta canary check: schema_version drifted")
if committed["contract_id"] != "psionic.train.sparta_canary.v1":
    fail("sparta canary check: contract_id drifted")

prereg = committed["preregistration"]
if "no public gradients into the main optimizer" not in prereg["standing_order"].lower():
    fail("sparta canary check: W3 standing order weakened or missing")
if prereg["baseline_arm"] != "synchronous_averaging":
    fail("sparta canary check: baseline arm must stay synchronous averaging")
if not prereg["grid"]:
    fail("sparta canary check: pre-registered grid is empty")
pluralis_start = prereg["grid"][0]
if pluralis_start["sparsity_fraction"] != 0.05 or pluralis_start["average_state_every_steps"] != 5:
    fail(
        "sparta canary check: grid must start from the Pluralis published tuning "
        "(sparsity 0.05, average every 5 steps)"
    )
for point in prereg["grid"]:
    if point["partition_strategy"] != "random_rotating_partitions":
        fail("sparta canary check: grid points must use random rotating partitions")
    if not (0.0 < point["sparsity_fraction"] <= 1.0) or point["average_state_every_steps"] < 1:
        fail("sparta canary check: grid point values out of range")

eval_schema = prereg["eval_schema"]
families = eval_schema["held_out_eval_families"]
if not families:
    fail("sparta canary check: held-out eval families must be nonempty")
if all("perplexity" in family.lower() for family in families):
    fail("sparta canary check: perplexity alone can never decide the canary")
if not eval_schema["first_divergence_step_metric"].strip():
    fail("sparta canary check: first-divergence-step metric must be named")

kill_bound = prereg["kill_bound"]
if not (0.0 < kill_bound["max_eval_degradation_fraction_vs_baseline"] < 1.0):
    fail("sparta canary check: eval-degradation kill bound out of range")
if not (0.0 < kill_bound["min_comms_savings_fraction_floor"] < 1.0):
    fail("sparta canary check: comms-savings materiality floor out of range")
digest = prereg["preregistration_digest"]
if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
    fail("sparta canary check: preregistration_digest is not a SHA256 hex digest")

artifact = committed["comparison_artifact"]
if artifact["measurement_status"] != "synthetic_bounded":
    fail(
        "sparta canary check: no real gated canary run exists; "
        "measurement_status must stay synthetic_bounded"
    )
if "synthetic" not in artifact["detail"].lower():
    fail("sparta canary check: synthetic bounded artifact must say it is synthetic")
averaging_rounds = [r for r in artifact["per_round"] if r["sparse_arm"]["averaged_this_round"]]
if not averaging_rounds:
    fail("sparta canary check: comparison must contain averaging rounds")
for round_entry in averaging_rounds:
    sparse = round_entry["sparse_arm"]
    sync = round_entry["synchronous_arm"]
    if sparse["simulated_bytes_communicated"] >= sync["simulated_bytes_communicated"]:
        fail("sparta canary check: sparse rounds must communicate fewer simulated bytes")
if artifact["total_sparse_bytes_communicated"] >= artifact["total_synchronous_bytes_communicated"]:
    fail("sparta canary check: sparse arm must communicate fewer total simulated bytes")
if artifact["averaging_rounds_to_full_coverage"] < 1:
    fail("sparta canary check: rounds-to-coverage must be at least 1")

outcome = committed["outcome_record"]
if outcome["status"] != "pending":
    fail(
        "sparta canary check: outcome must stay pending until a real gated canary run "
        "produces a held/killed decision"
    )
if outcome["decided_by"] is not None:
    fail("sparta canary check: pending outcomes carry no deciding bound")
if outcome["preregistration_digest"] != digest:
    fail("sparta canary check: outcome record is not bound to the pre-registration digest")
if outcome["derisking_ledger_path"] != "docs/PSION_DERISKING_LEDGER.md":
    fail("sparta canary check: derisking ledger path drifted")
if outcome["derisking_ledger_entry"] != "SPARTA-class sparse parameter averaging":
    fail("sparta canary check: derisking ledger entry name drifted")

if "synthetic" not in committed["claim_boundary"].lower():
    fail("sparta canary check: claim_boundary must keep the synthetic boundary explicit")
if "no public gradients into the main optimizer" not in committed["claim_boundary"].lower():
    fail("sparta canary check: claim_boundary must restate the W3 standing order")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "preregistration_digest": digest,
    "grid_point_count": len(prereg["grid"]),
    "averaging_rounds_to_full_coverage": artifact["averaging_rounds_to_full_coverage"],
    "total_sparse_bytes_communicated": artifact["total_sparse_bytes_communicated"],
    "total_synchronous_bytes_communicated": artifact["total_synchronous_bytes_communicated"],
    "outcome_status": outcome["status"],
    "measurement_status": artifact["measurement_status"],
}
print(json.dumps(summary, indent=2))
PY
