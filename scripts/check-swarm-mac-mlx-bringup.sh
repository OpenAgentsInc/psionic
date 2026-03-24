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
if report["training_backend_posture"] != "missing_open_adapter_backend":
    fail("swarm mac mlx bring-up error: backend posture should stay missing_open_adapter_backend before SWARM-4 lands")
if "open-adapter backend is not implemented yet" not in report.get("training_backend_blocker", ""):
    fail("swarm mac mlx bring-up error: training_backend_blocker must make the missing backend explicit")
if "dense_f32.matmul" not in report["admitted_metal_slice"]:
    fail("swarm mac mlx bring-up error: admitted_metal_slice must include dense_f32.matmul")
if report["finished_at_ms"] < report["started_at_ms"]:
    fail("swarm mac mlx bring-up error: finished_at_ms must not be earlier than started_at_ms")
if report["observed_wallclock_ms"] <= 0:
    fail("swarm mac mlx bring-up error: observed_wallclock_ms must be positive")
if not isinstance(report["drift_notes"], list) or not report["drift_notes"]:
    fail("swarm mac mlx bring-up error: drift_notes must be a non-empty list")
if "does not claim the Mac host already has a finished MLX open-adapter training backend" not in report["claim_boundary"]:
    fail("swarm mac mlx bring-up error: claim boundary drifted")

summary = {
    "verdict": "verified",
    "disposition": report["disposition"],
    "training_backend_posture": report["training_backend_posture"],
    "chip_name": host["chip_name"],
    "metal_family_support": host.get("metal_family_support"),
    "machine_contract_satisfied": report["machine_contract_satisfied"],
}
print(json.dumps(summary, indent=2))
PY
