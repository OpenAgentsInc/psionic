#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/internet_fault_harness_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/internet_fault_harness_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/internet_fault_harness_contract_v1.json"
cargo run -q -p psionic-train --bin internet_fault_harness_contract -- "${generated_path}" >/dev/null

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
    fail("internet fault harness contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.internet_fault_harness_contract.v1":
    fail("internet fault harness contract check: schema_version drifted")
if committed["contract_id"] != "psionic.internet_fault_harness_contract.v1":
    fail("internet fault harness contract check: contract_id drifted")
if len(committed["fault_profiles"]) != 4:
    fail("internet fault harness contract check: expected four fault profiles")
if len(committed["throughput_baselines"]) != 3:
    fail("internet fault harness contract check: expected three throughput baselines")
if len(committed["soak_suites"]) != 2:
    fail("internet fault harness contract check: expected two soak suites")
if len(committed["run_receipts"]) != 7:
    fail("internet fault harness contract check: expected seven run receipts")

held = [receipt for receipt in committed["run_receipts"] if receipt["disposition"] == "held"]
if len(held) != 1:
    fail("internet fault harness contract check: expected one held validator-loss receipt")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "suite_ids": [suite["suite_id"] for suite in committed["soak_suites"]],
}
print(json.dumps(summary, indent=2))
PY
