#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/validator_challenge_scoring_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/validator_challenge_scoring_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/validator_challenge_scoring_contract_v1.json"
cargo run -q -p psionic-train --bin validator_challenge_scoring_contract -- "${generated_path}" >/dev/null

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
    fail("validator challenge scoring contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.validator_challenge_scoring_contract.v1":
    fail("validator challenge scoring contract check: schema_version drifted")
if committed["contract_id"] != "psionic.validator_challenge_scoring_contract.v1":
    fail("validator challenge scoring contract check: contract_id drifted")
if len(committed["replay_rules"]) != 2:
    fail("validator challenge scoring contract check: expected two replay rules")
if len(committed["score_receipts"]) != 2:
    fail("validator challenge scoring contract check: expected two score receipts")
if len(committed["refusals"]) != 1:
    fail("validator challenge scoring contract check: expected one refusal")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "score_receipt_ids": [receipt["receipt_id"] for receipt in committed["score_receipts"]],
}
print(json.dumps(summary, indent=2))
PY
