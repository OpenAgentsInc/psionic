#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/quantized_outer_sync_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/quantized_outer_sync_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/quantized_outer_sync_contract_v1.json"
cargo run -q -p psionic-train --bin quantized_outer_sync_contract -- "${generated_path}" >/dev/null

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
    fail("quantized outer sync contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.quantized_outer_sync_contract.v1":
    fail("quantized outer sync contract check: schema_version drifted")
if committed["contract_id"] != "psionic.quantized_outer_sync_contract.v1":
    fail("quantized outer sync contract check: contract_id drifted")
if len(committed["delta_policies"]) != 3:
    fail("quantized outer sync contract check: expected three delta policies")
if len(committed["exchange_receipts"]) != 3:
    fail("quantized outer sync contract check: expected three exchange receipts")
if len(committed["aggregation_receipts"]) != 1:
    fail("quantized outer sync contract check: expected one aggregation receipt")
if len(committed["correctness_receipts"]) != 2:
    fail("quantized outer sync contract check: expected two correctness receipts")

dispositions = {receipt["disposition"] for receipt in committed["exchange_receipts"]}
if dispositions != {"applied", "refused"}:
    fail("quantized outer sync contract check: expected both applied and refused exchanges")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "exchange_ids": [receipt["receipt_id"] for receipt in committed["exchange_receipts"]],
}
print(json.dumps(summary, indent=2))
PY
