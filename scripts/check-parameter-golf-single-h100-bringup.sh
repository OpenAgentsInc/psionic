#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
input_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_single_h100_bringup.json"
dataset_root="${HOME}/code/parameter-golf/data/datasets/fineweb10B_sp1024"
tokenizer_path="${HOME}/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
report_path=""

usage() {
    cat <<'EOF' >&2
Usage: scripts/check-parameter-golf-single-h100-bringup.sh [--input <path>] [--dataset-root <path>] [--tokenizer-path <path>] [--report <path>]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 ]] || {
                echo "missing path after --input" >&2
                usage
                exit 1
            }
            input_path="$2"
            shift 2
            ;;
        --dataset-root)
            [[ $# -ge 2 ]] || {
                echo "missing path after --dataset-root" >&2
                usage
                exit 1
            }
            dataset_root="$2"
            shift 2
            ;;
        --tokenizer-path)
            [[ $# -ge 2 ]] || {
                echo "missing path after --tokenizer-path" >&2
                usage
                exit 1
            }
            tokenizer_path="$2"
            shift 2
            ;;
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
    report_path="$(mktemp "${TMPDIR:-/tmp}/parameter_golf_single_h100_bringup.XXXXXX.json")"
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

cargo run -q -p psionic-train --bin parameter_golf_single_h100_bringup -- \
    "${dataset_root}" \
    "${tokenizer_path}" \
    "${report_path}"

python3 - "${input_path}" "${report_path}" "${dataset_root}" "${tokenizer_path}" <<'PY'
import json
import os
import sys
from pathlib import Path

expected_path = Path(os.path.expanduser(sys.argv[1]))
actual_path = Path(os.path.expanduser(sys.argv[2]))
dataset_root = os.path.expanduser(sys.argv[3])
tokenizer_path = os.path.expanduser(sys.argv[4])


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


expected = json.loads(expected_path.read_text(encoding="utf-8"))
actual = json.loads(actual_path.read_text(encoding="utf-8"))

for path in (dataset_root, tokenizer_path):
    if not os.path.exists(path):
        fail(f"parameter golf single-H100 bring-up error: missing required local path `{path}`")

exact_keys = [
    "schema_version",
    "scope_window",
    "dataset_key",
    "variant",
    "tokenizer_digest",
    "dataset_manifest_digest",
    "train_shard_count",
    "validation_shard_count",
    "train_token_count",
    "validation_token_count",
    "train_selection_posture",
    "validation_identity",
    "geometry",
    "hyperparameters",
    "machine_thresholds",
    "baseline_model_id",
    "baseline_model_revision",
    "baseline_model_config",
    "baseline_model_descriptor_digest",
    "optimizer_plan_digest",
    "cuda_training_capability_report_digest",
    "challenge_kernel_blockers",
    "execution_posture",
    "disposition",
]
for key in exact_keys:
    if actual.get(key) != expected.get(key):
        fail(
            f"parameter golf single-H100 bring-up error: `{key}` drifted\nexpected: {expected.get(key)!r}\nactual:   {actual.get(key)!r}"
        )

if actual.get("dataset_root") != dataset_root:
    fail("parameter golf single-H100 bring-up error: dataset_root does not match the requested path")
if actual.get("tokenizer_path") != tokenizer_path:
    fail("parameter golf single-H100 bring-up error: tokenizer_path does not match the requested path")
if actual.get("psionic_entrypoint") != (
    f"cargo run -q -p psionic-train --bin parameter_golf_single_h100_bringup -- {dataset_root} {tokenizer_path}"
):
    fail("parameter golf single-H100 bring-up error: unexpected psionic entrypoint")
if "torchrun --standalone --nproc_per_node=1 train_gpt.py" not in actual.get("upstream_reference_entrypoint", ""):
    fail("parameter golf single-H100 bring-up error: upstream reference entrypoint does not point at the public single-H100 command")
if actual.get("matching_h100_device_count") != expected.get("matching_h100_device_count"):
    fail("parameter golf single-H100 bring-up error: matching_h100_device_count drifted")
if actual.get("machine_contract_satisfied") != expected.get("machine_contract_satisfied"):
    fail("parameter golf single-H100 bring-up error: machine_contract_satisfied drifted")
if actual.get("refusal", {}).get("subject") != expected.get("refusal", {}).get("subject"):
    fail("parameter golf single-H100 bring-up error: primary refusal subject drifted")
if actual.get("cuda_blocker_refusal", {}).get("subject") != "parameter_golf_cuda_training":
    fail("parameter golf single-H100 bring-up error: cuda_blocker_refusal should stay bound to parameter_golf_cuda_training")
if actual.get("observed_cuda_devices") != expected.get("observed_cuda_devices"):
    fail("parameter golf single-H100 bring-up error: observed_cuda_devices drifted")
if actual.get("observed_cuda_health", {}).get("status") != expected.get("observed_cuda_health", {}).get("status"):
    fail("parameter golf single-H100 bring-up error: observed_cuda_health.status drifted")
if actual.get("final_val_loss") is not None:
    fail("parameter golf single-H100 bring-up error: final_val_loss must stay absent while training is not executed")
if actual.get("final_val_bpb") is not None:
    fail("parameter golf single-H100 bring-up error: final_val_bpb must stay absent while training is not executed")
if actual.get("compressed_model_bytes") is not None:
    fail("parameter golf single-H100 bring-up error: compressed_model_bytes must stay absent while training is not executed")
if not isinstance(actual.get("drift_notes"), list) or not actual["drift_notes"]:
    fail("parameter golf single-H100 bring-up error: drift_notes must be a non-empty list")
if actual.get("observed_wallclock_ms", 0) <= 0:
    fail("parameter golf single-H100 bring-up error: observed_wallclock_ms must be positive")
if actual.get("finished_at_ms", 0) < actual.get("started_at_ms", 0):
    fail("parameter golf single-H100 bring-up error: finished_at_ms must not be earlier than started_at_ms")

summary = {
    "verdict": "verified",
    "dataset_manifest_digest": actual["dataset_manifest_digest"],
    "tokenizer_digest": actual["tokenizer_digest"]["tokenizer_digest"],
    "disposition": actual["disposition"],
    "matching_h100_device_count": actual["matching_h100_device_count"],
    "machine_contract_satisfied": actual["machine_contract_satisfied"],
    "execution_posture": actual["execution_posture"],
    "observed_wallclock_ms": actual["observed_wallclock_ms"],
}
print(json.dumps(summary, indent=2))
PY
