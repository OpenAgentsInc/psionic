#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/public_dataset_authority_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/public_dataset_authority_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/public_dataset_authority_contract_v1.json"
cargo run -q -p psionic-train --bin public_dataset_authority_contract -- "${generated_path}" >/dev/null

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
    fail("public dataset authority contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.public_dataset_authority_contract.v1":
    fail("public dataset authority contract check: schema_version drifted")
if committed["contract_id"] != "psionic.public_dataset_authority_contract.v1":
    fail("public dataset authority contract check: contract_id drifted")
if len(committed["dataset_pages"]) != 8:
    fail("public dataset authority contract check: expected eight dataset pages")
if len(committed["page_proofs"]) != 8:
    fail("public dataset authority contract check: expected eight page proofs")
if len(committed["anti_replay_receipts"]) != 5:
    fail("public dataset authority contract check: expected five anti-replay receipts")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "duplicate_receipt": [
        receipt["receipt_id"]
        for receipt in committed["anti_replay_receipts"]
        if receipt["disposition"] == "refused_duplicate"
    ],
}
print(json.dumps(summary, indent=2))
PY
