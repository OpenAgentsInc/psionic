#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_path=""

usage() {
    cat <<'EOF' >&2
Usage: scripts/check-swarm-mac-mlx-bringup.sh [--report <path>]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --report)
            [[ $# -ge 2 ]] || {
                echo "missing path after --report" >&2
                usage
                exit 1
            }
            report_path="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${report_path}" ]]; then
    report_path="$(mktemp "${TMPDIR:-/tmp}/swarm_mac_mlx_bringup.XXXXXX.json")"
    cleanup_report=1
else
    cleanup_report=0
fi

cleanup() {
    if [[ "${cleanup_report}" -eq 1 ]]; then
        rm -f -- "${report_path}"
    fi
}
trap cleanup EXIT

cargo run -q -p psionic-train --bin swarm_mac_mlx_bringup -- "${report_path}"

python3 - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


required_top_level = [
    "schema_version",
    "scope_window",
    "run_family_id",
    "contract_digest",
    "host",
    "machine_thresholds",
    "machine_contract_satisfied",
    "admitted_metal_slice",
    "training_backend_posture",
    "disposition",
    "psionic_entrypoint",
    "claim_boundary",
    "drift_notes",
    "report_digest",
]
for key in required_top_level:
    if key not in report:
        fail(f"swarm mac mlx bring-up error: missing required top-level field `{key}`")

host = report["host"]
for key in [
    "hostname",
    "os_name",
    "os_version",
    "os_build_version",
    "architecture",
    "hardware_model",
    "chip_name",
    "unified_memory_bytes",
]:
    if key not in host:
        fail(f"swarm mac mlx bring-up error: missing host field `{key}`")

thresholds = report["machine_thresholds"]
if thresholds["precision_policy"] != "f32_reference":
    fail("swarm mac mlx bring-up error: precision_policy must stay f32_reference for the first lane")
if thresholds["required_backend_label"] != "open_adapter_backend.mlx.metal.gpt_oss_lm_head":
    fail("swarm mac mlx bring-up error: required backend label drifted")
if thresholds["safe_sequence_length_tokens"] <= 0 or thresholds["safe_microbatch_size"] <= 0:
    fail("swarm mac mlx bring-up error: safe geometry must stay positive")
if report["training_backend_posture"] != "ready":
    fail("swarm mac mlx bring-up error: backend posture must be ready after SWARM-4 lands")
if report["disposition"] != "ready_to_attempt":
    fail("swarm mac mlx bring-up error: disposition must be ready_to_attempt when the gate succeeds")
if report.get("training_backend_blocker") is not None:
    fail("swarm mac mlx bring-up error: training_backend_blocker must be absent once the gate succeeds")
overfit_gate = report.get("overfit_gate")
if not isinstance(overfit_gate, dict):
    fail("swarm mac mlx bring-up error: overfit_gate must be present once the MLX backend lands")
for key in [
    "run_id",
    "execution_backend_label",
    "logical_device_kind",
    "logical_device_label",
    "adapter_family",
    "precision_policy",
    "executed_steps",
    "batch_count",
    "final_mean_loss",
    "adapter_artifact_digest",
    "adapter_identity_digest",
    "execution_provenance_digest",
    "final_state_dict_digest",
    "probe_top_token_id",
    "unsupported_precision_refusal",
    "gate_digest",
]:
    if key not in overfit_gate:
        fail(f"swarm mac mlx bring-up error: missing overfit_gate field `{key}`")
if overfit_gate["execution_backend_label"] != "open_adapter_backend.mlx.metal.gpt_oss_lm_head":
    fail("swarm mac mlx bring-up error: overfit gate backend label drifted")
if overfit_gate["logical_device_kind"] != "metal":
    fail("swarm mac mlx bring-up error: overfit gate must surface logical_device_kind=metal")
if overfit_gate["logical_device_label"] != "metal:0":
    fail("swarm mac mlx bring-up error: overfit gate must surface logical_device_label=metal:0")
if overfit_gate["adapter_family"] != "gpt_oss.decoder_lm_head_lora":
    fail("swarm mac mlx bring-up error: overfit gate adapter family drifted")
if overfit_gate["precision_policy"] != "f32_reference":
    fail("swarm mac mlx bring-up error: overfit gate precision policy must stay f32_reference")
if overfit_gate["executed_steps"] <= 0 or overfit_gate["batch_count"] <= 0:
    fail("swarm mac mlx bring-up error: overfit gate must execute at least one step and one batch")
if overfit_gate["final_mean_loss"] <= 0:
    fail("swarm mac mlx bring-up error: overfit gate final_mean_loss must be positive")
if overfit_gate["probe_top_token_id"] != 2:
    fail("swarm mac mlx bring-up error: deterministic probe must target token 2")
if "does not yet support precision policy" not in overfit_gate["unsupported_precision_refusal"]:
    fail("swarm mac mlx bring-up error: overfit gate must retain explicit unsupported precision refusal")
if "dense_f32.matmul" not in report["admitted_metal_slice"]:
    fail("swarm mac mlx bring-up error: admitted_metal_slice must include dense_f32.matmul")
if report["finished_at_ms"] < report["started_at_ms"]:
    fail("swarm mac mlx bring-up error: finished_at_ms must not be earlier than started_at_ms")
if report["observed_wallclock_ms"] <= 0:
    fail("swarm mac mlx bring-up error: observed_wallclock_ms must be positive")
if not isinstance(report["drift_notes"], list) or not report["drift_notes"]:
    fail("swarm mac mlx bring-up error: drift_notes must be a non-empty list")
if "one bounded local open-adapter run that emits a real adapter artifact under the MLX Metal backend label" not in report["claim_boundary"]:
    fail("swarm mac mlx bring-up error: claim boundary drifted")

summary = {
    "verdict": "verified",
    "disposition": report["disposition"],
    "training_backend_posture": report["training_backend_posture"],
    "overfit_gate_run_id": overfit_gate["run_id"],
    "probe_top_token_id": overfit_gate["probe_top_token_id"],
    "chip_name": host["chip_name"],
    "metal_family_support": host.get("metal_family_support"),
    "machine_contract_satisfied": report["machine_contract_satisfied"],
}
print(json.dumps(summary, indent=2))
PY
