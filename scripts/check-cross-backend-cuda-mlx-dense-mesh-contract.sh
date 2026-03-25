#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/cross_backend_cuda_mlx_dense_mesh_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/cross_backend_cuda_mlx_dense.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/cross_backend_cuda_mlx_dense_mesh_contract_v1.json"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-}" cargo run -q -p psionic-train --bin cross_backend_cuda_mlx_dense_mesh_contract -- "${generated_path}" >/dev/null

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
    fail("cross-backend CUDA+MLX dense mesh check: committed fixture drifted from generator output")

participants = {participant["backend_family"] for participant in committed["participant_backends"]}
if participants != {"cuda", "mlx_metal"}:
    fail("cross-backend CUDA+MLX dense mesh check: participant backend set drifted")

collective = committed["collective_contract"]
if collective["gradient_collective_kind"] != "all_reduce" or collective["master_weight_collective_kind"] != "broadcast":
    fail("cross-backend CUDA+MLX dense mesh check: collective kinds drifted")
if collective["reduction_precision"] != "fp32" or collective["communication_quantization"] != "none":
    fail("cross-backend CUDA+MLX dense mesh check: collective precision drifted")

precision = committed["precision_policy"]
expected_precision = {
    "parameter_precision": "fp32",
    "gradient_precision": "fp32",
    "optimizer_state_precision": "fp32",
    "master_weight_precision": "fp32",
    "reduction_precision": "fp32",
    "communication_quantization": "none",
    "stochastic_rounding": False,
    "loss_scale": None,
}
for key, expected in expected_precision.items():
    if precision.get(key) != expected:
        fail(f"cross-backend CUDA+MLX dense mesh check: precision field {key} drifted")

refusals = {refusal["refusal_kind"] for refusal in committed["refusal_set"]}
expected_refusals = {
    "bf16_mixed_precision",
    "fp16_dynamic_loss_scaling",
    "direct_nccl_participation_by_mlx_rank",
    "split_master_weight_authority",
    "checkpointless_optimizer_migration",
}
if refusals != expected_refusals:
    fail("cross-backend CUDA+MLX dense mesh check: refusal set drifted")

summary = {
    "verdict": "verified",
    "participant_count": len(committed["participant_backends"]),
    "optimizer_kind": committed["optimizer_ownership"]["optimizer_kind"],
    "distributed_optimizer_kind": committed["optimizer_ownership"]["distributed_optimizer_kind"],
}
print(json.dumps(summary, indent=2))
PY
