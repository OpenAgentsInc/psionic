#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/public_miner_protocol_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/public_miner_protocol_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/public_miner_protocol_contract_v1.json"
cargo run -q -p psionic-train --bin public_miner_protocol_contract -- "${generated_path}" >/dev/null

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
    fail("public miner protocol contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.public_miner_protocol_contract.v1":
    fail("public miner protocol contract check: schema_version drifted")
if committed["contract_id"] != "psionic.public_miner_protocol_contract.v1":
    fail("public miner protocol contract check: contract_id drifted")
if len(committed["sessions"]) != 2:
    fail("public miner protocol contract check: expected two active sessions")
if len(committed["local_step_receipts"]) != 2:
    fail("public miner protocol contract check: expected two local-step receipts")
if len(committed["delta_upload_receipts"]) != 2:
    fail("public miner protocol contract check: expected two delta-upload receipts")
if len(committed["checkpoint_sync_receipts"]) != 2:
    fail("public miner protocol contract check: expected two checkpoint-sync receipts")
if len(committed["refusals"]) != 1:
    fail("public miner protocol contract check: expected one refusal")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "refusal_ids": [refusal["refusal_id"] for refusal in committed["refusals"]],
}
print(json.dumps(summary, indent=2))
PY
