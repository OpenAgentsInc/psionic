#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/mixed_backend_checkpoint_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/mixed_backend_checkpoint.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/mixed_backend_checkpoint_contract_v1.json"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-}" cargo run -q -p psionic-train --bin mixed_backend_checkpoint_contract -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

committed = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))
if committed != generated:
    fail("mixed-backend checkpoint check: committed fixture drifted from generator output")

if committed["checkpoint_manifest"]["checkpoint_family"] != "psion.cross_provider.pretrain.v1":
    fail("mixed-backend checkpoint check: checkpoint family drifted")
if len(committed["checkpoint_manifest"]["shards"]) != 4:
    fail("mixed-backend checkpoint check: shard count drifted")

backend_receipts = {receipt["backend_family"] for receipt in committed["portable_state_receipts"]}
if backend_receipts != {"cuda", "mlx_metal"}:
    fail("mixed-backend checkpoint check: portable state receipt backends drifted")

for receipt in committed["portable_state_receipts"]:
    if receipt["artifact_format"] != "safetensors":
        fail("mixed-backend checkpoint check: portable state artifact format drifted")
    if receipt["parameter_precision"] != "fp32" or receipt["optimizer_state_precision"] != "fp32":
        fail("mixed-backend checkpoint check: portable state precision drifted")

restore_dispositions = {receipt["disposition"] for receipt in committed["restore_receipts"]}
if restore_dispositions != {"recovered", "restored_with_canonical_cast", "refused"}:
    fail("mixed-backend checkpoint check: restore disposition set drifted")

refusals = {refusal["refusal_kind"] for refusal in committed["refusal_set"]}
expected_refusals = {
    "bf16_optimizer_state_migration",
    "quantized_checkpoint_resume",
    "checkpointless_migration",
    "incomplete_portable_group_selection",
}
if refusals != expected_refusals:
    fail("mixed-backend checkpoint check: refusal set drifted")

summary = {
    "verdict": "verified",
    "portable_state_receipt_count": len(committed["portable_state_receipts"]),
    "restore_receipt_count": len(committed["restore_receipts"]),
    "refusal_count": len(committed["refusal_set"]),
}
print(json.dumps(summary, indent=2))
PY
