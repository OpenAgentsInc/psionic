#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
inventory_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_single_h100_bringup.json"
report_path=""

usage() {
    cat <<'EOF' >&2
Usage: scripts/check-swarm-linux-4080-bringup.sh [--inventory <path>] [--report <path>]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inventory)
            [[ $# -ge 2 ]] || {
                echo "missing path after --inventory" >&2
                usage
                exit 1
            }
            inventory_path="$2"
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
    report_path="$(mktemp "${TMPDIR:-/tmp}/swarm_linux_4080_bringup.XXXXXX.json")"
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

cargo run -q -p psionic-train --bin swarm_linux_cuda_bringup -- "${inventory_path}" "${report_path}"

python3 - "${report_path}" "${inventory_path}" <<'PY'
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
inventory_path = os.path.expanduser(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if report["source_inventory_report_path"] != inventory_path:
    fail("swarm linux 4080 bring-up error: source inventory path does not match the requested input")
if report["machine_thresholds"]["required_backend"] != "cuda":
    fail("swarm linux 4080 bring-up error: required backend drifted")
if report["machine_thresholds"]["required_device_name"] != "RTX 4080":
    fail("swarm linux 4080 bring-up error: required device name drifted")
if report["machine_thresholds"]["precision_policy"] != "f32_reference":
    fail("swarm linux 4080 bring-up error: precision policy must stay f32_reference")
if report["matching_rtx4080_device_count"] < 1:
    fail("swarm linux 4080 bring-up error: expected at least one matching RTX 4080 device")
if not report["machine_contract_satisfied"]:
    fail("swarm linux 4080 bring-up error: retained inventory should satisfy the RTX 4080 contract")
if report["disposition"] != "ready_to_attempt":
    fail("swarm linux 4080 bring-up error: disposition should stay ready_to_attempt")

harness = report["parity_harness"]
if harness["execution_backend_label"] != "open_adapter_backend.cuda.gpt_oss_lm_head":
    fail("swarm linux 4080 bring-up error: parity harness backend label drifted")
if harness["adapter_family"] != "gpt_oss.decoder_lm_head_lora":
    fail("swarm linux 4080 bring-up error: parity harness adapter family drifted")
if harness["precision_policy"] != "f32_reference":
    fail("swarm linux 4080 bring-up error: parity harness precision policy drifted")
if harness["executed_steps"] <= 0 or harness["batch_count"] <= 0:
    fail("swarm linux 4080 bring-up error: parity harness must execute at least one step and batch")
if harness["final_mean_loss"] <= 0:
    fail("swarm linux 4080 bring-up error: parity harness final_mean_loss must be positive")
if "does not yet support precision policy" not in harness["unsupported_precision_refusal"]:
    fail("swarm linux 4080 bring-up error: parity harness must keep unsupported precision refusal explicit")
contributor_receipt = harness.get("contributor_receipt")
if not isinstance(contributor_receipt, dict):
    fail("swarm linux 4080 bring-up error: parity harness must carry contributor_receipt")
for key in [
    "schema_version",
    "manifest",
    "run_id",
    "checkpoint_family",
    "executed_steps",
    "batch_count",
    "final_mean_loss",
    "adapter_artifact_digest",
    "adapter_identity_digest",
    "initial_state_dict_digest",
    "final_state_dict_digest",
    "execution_provenance_digest",
    "deterministic_probe_top_token_id",
    "unsupported_precision_refusal",
    "receipt_digest",
]:
    if key not in contributor_receipt:
        fail(f"swarm linux 4080 bring-up error: missing contributor_receipt field `{key}`")
manifest = contributor_receipt["manifest"]
for key in [
    "receipt_contract_digest",
    "run_family_id",
    "swarm_contract_digest",
    "dataset_ref",
    "dataset_manifest_digest",
    "validator_policy_id",
    "aggregation_policy_id",
    "replay_policy_id",
    "contributor_role_id",
    "execution_backend_label",
    "logical_device_kind",
    "logical_device_label",
    "admissible_model_family",
    "adapter_family",
    "adapter_format",
    "precision_policy",
    "base_model_id",
    "base_model_revision",
    "base_served_artifact_digest",
    "tokenizer_digest",
    "tokenizer_contract_digest",
    "hidden_size",
    "vocab_size",
    "lora_rank",
    "lora_alpha",
    "batch_size",
    "sample_count",
    "shared_replay_identity_digest",
    "manifest_digest",
]:
    if key not in manifest:
        fail(f"swarm linux 4080 bring-up error: missing contributor_receipt.manifest field `{key}`")
if contributor_receipt["schema_version"] != "swarm.open_adapter.contributor_receipt.v1":
    fail("swarm linux 4080 bring-up error: contributor receipt schema version drifted")
if manifest["contributor_role_id"] != "swarm.linux.cuda.rtx4080.contributor":
    fail("swarm linux 4080 bring-up error: contributor receipt role drifted")
if manifest["execution_backend_label"] != harness["execution_backend_label"]:
    fail("swarm linux 4080 bring-up error: contributor receipt backend label must match the parity harness backend label")
if manifest["logical_device_kind"] != "cuda" or manifest["logical_device_label"] != "cuda:0":
    fail("swarm linux 4080 bring-up error: contributor receipt must preserve the CUDA logical-device identity")
if manifest["adapter_family"] != "gpt_oss.decoder_lm_head_lora" or manifest["adapter_format"] != "safetensors":
    fail("swarm linux 4080 bring-up error: contributor receipt adapter identity drifted")
if manifest["precision_policy"] != "f32_reference":
    fail("swarm linux 4080 bring-up error: contributor receipt precision policy must stay f32_reference")
if contributor_receipt["deterministic_probe_top_token_id"] != harness["probe_top_token_id"]:
    fail("swarm linux 4080 bring-up error: contributor receipt probe token must match the parity harness probe token")
if contributor_receipt["unsupported_precision_refusal"] != harness["unsupported_precision_refusal"]:
    fail("swarm linux 4080 bring-up error: contributor receipt unsupported precision refusal must match the parity harness refusal")
if report["finished_at_ms"] < report["started_at_ms"]:
    fail("swarm linux 4080 bring-up error: finished_at_ms must not be earlier than started_at_ms")
if report["observed_wallclock_ms"] <= 0:
    fail("swarm linux 4080 bring-up error: observed_wallclock_ms must be positive")

summary = {
    "verdict": "verified",
    "disposition": report["disposition"],
    "matching_rtx4080_device_count": report["matching_rtx4080_device_count"],
    "backend_label": harness["execution_backend_label"],
    "probe_top_token_id": harness["probe_top_token_id"],
    "receipt_contract_digest": manifest["receipt_contract_digest"],
}
print(json.dumps(summary, indent=2))
PY
