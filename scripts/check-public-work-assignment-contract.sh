#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/public_work_assignment_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/public_work_assignment_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/public_work_assignment_contract_v1.json"
cargo run -q -p psionic-train --bin public_work_assignment_contract -- "${generated_path}" >/dev/null

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
    fail("public work assignment contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.public_work_assignment_contract.v1":
    fail("public work assignment contract check: schema_version drifted")
if committed["contract_id"] != "psionic.public_work_assignment_contract.v1":
    fail("public work assignment contract check: contract_id drifted")
if len(committed["windows"]) != 2:
    fail("public work assignment contract check: expected two windows")
if len(committed["assignments"]) != 8:
    fail("public work assignment contract check: expected eight assignments")
if len(committed["assignment_receipts"]) != 8:
    fail("public work assignment contract check: expected eight assignment receipts")
if len(committed["late_window_refusals"]) != 1:
    fail("public work assignment contract check: expected one late-window refusal")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "window_ids": [window["window_id"] for window in committed["windows"]],
}
print(json.dumps(summary, indent=2))
PY
