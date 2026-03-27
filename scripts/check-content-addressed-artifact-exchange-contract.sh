#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/content_addressed_artifact_exchange_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/content_addressed_artifact_exchange_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/content_addressed_artifact_exchange_contract_v1.json"
cargo run -q -p psionic-train --bin content_addressed_artifact_exchange_contract -- "${generated_path}" >/dev/null

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
    fail("content-addressed artifact exchange contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.content_addressed_artifact_exchange_contract.v1":
    fail("content-addressed artifact exchange contract check: schema_version drifted")
if committed["contract_id"] != "psionic.content_addressed_artifact_exchange_contract.v1":
    fail("content-addressed artifact exchange contract check: contract_id drifted")
if len(committed["exchange_backends"]) != 5:
    fail("content-addressed artifact exchange contract check: expected five exchange backends")
if len(committed["published_artifacts"]) != 5:
    fail("content-addressed artifact exchange contract check: expected five published artifacts")
if len(committed["fetch_receipts"]) != 5:
    fail("content-addressed artifact exchange contract check: expected five fetch receipts")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "refused_fetches": [
        receipt["receipt_id"]
        for receipt in committed["fetch_receipts"]
        if receipt["disposition"] == "refused"
    ],
}
print(json.dumps(summary, indent=2))
PY
