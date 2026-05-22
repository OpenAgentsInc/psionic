#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/multi_validator_consensus_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/multi_validator_consensus_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/multi_validator_consensus_contract_v1.json"
cargo run -q -p psionic-train --bin multi_validator_consensus_contract -- "${generated_path}" >/dev/null

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
    fail("multi-validator consensus contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.multi_validator_consensus_contract.v1":
    fail("multi-validator consensus contract check: schema_version drifted")
if committed["contract_id"] != "psionic.multi_validator_consensus_contract.v1":
    fail("multi-validator consensus contract check: contract_id drifted")
if len(committed["votes"]) != 2:
    fail("multi-validator consensus contract check: expected two votes")
if len(committed["promotion_decisions"]) != 1:
    fail("multi-validator consensus contract check: expected one promotion decision")
if len(committed["disagreement_receipts"]) != 1:
    fail("multi-validator consensus contract check: expected one disagreement receipt")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "decision_ids": [decision["decision_id"] for decision in committed["promotion_decisions"]],
}
print(json.dumps(summary, indent=2))
PY
