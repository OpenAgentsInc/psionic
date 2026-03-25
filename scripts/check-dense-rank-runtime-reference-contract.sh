#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/dense_rank_runtime_reference_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/dense_rank_runtime_reference_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/dense_rank_runtime_reference_contract_v1.json"
cargo run -q -p psionic-train --bin dense_rank_runtime_reference_contract -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if fixture != generated:
    fail("dense-rank runtime check: committed fixture drifted from generator output")

runtime = fixture["runtime"]
if runtime["runtime_family_id"] != "psionic.dense_rank_runtime.cuda.v1":
    fail("dense-rank runtime check: runtime family drifted")
if runtime["consumer_lane_id"] != "psion.cross_provider_pretraining_dense_reference":
    fail("dense-rank runtime check: reference consumer lane drifted")
if runtime["requested_backend"] != "nccl":
    fail("dense-rank runtime check: requested backend drifted")
if runtime["world_size"] != 8:
    fail("dense-rank runtime check: world size drifted")
if fixture["generic_execution_receipt_name"] != "dense_rank_runtime_execution_receipt_v1.json":
    fail("dense-rank runtime check: execution receipt name drifted")
if fixture["validation_hook"]["hook_id"] != "dense_rank_runtime.validation.post_train_eval":
    fail("dense-rank runtime check: validation hook drifted")
if fixture["checkpoint_hook"]["hook_id"] != "dense_rank_runtime.checkpoint.materialization":
    fail("dense-rank runtime check: checkpoint hook drifted")

summary = {
    "verdict": "verified",
    "runtime_family_id": runtime["runtime_family_id"],
    "consumer_lane_id": runtime["consumer_lane_id"],
    "world_size": runtime["world_size"],
}
print(json.dumps(summary, indent=2))
PY
