#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/live_checkpoint_catchup_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/live_checkpoint_catchup_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/live_checkpoint_catchup_contract_v1.json"
cargo run -q -p psionic-train --bin live_checkpoint_catchup_contract -- "${generated_path}" >/dev/null

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
    fail("live checkpoint catchup contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.live_checkpoint_catchup_contract.v1":
    fail("live checkpoint catchup contract check: schema_version drifted")
if committed["contract_id"] != "psionic.live_checkpoint_catchup_contract.v1":
    fail("live checkpoint catchup contract check: contract_id drifted")
if len(committed["advertisements"]) != 3:
    fail("live checkpoint catchup contract check: expected three advertisements")
if len(committed["resume_windows"]) != 2:
    fail("live checkpoint catchup contract check: expected two resume windows")
if len(committed["catchup_receipts"]) != 2:
    fail("live checkpoint catchup contract check: expected two catchup receipts")

completed = [receipt for receipt in committed["catchup_receipts"] if receipt["disposition"] == "completed"]
refused = [receipt for receipt in committed["catchup_receipts"] if receipt["disposition"] == "refused"]
if len(completed) != 1 or len(refused) != 1:
    fail("live checkpoint catchup contract check: expected one completed and one refused receipt")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "receipt_ids": [receipt["receipt_id"] for receipt in committed["catchup_receipts"]],
}
print(json.dumps(summary, indent=2))
PY
