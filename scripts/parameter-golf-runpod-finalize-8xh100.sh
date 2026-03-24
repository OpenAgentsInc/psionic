#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  if [[ -d "${HOME}/.cargo/bin" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required but was not found in PATH or \${HOME}/.cargo/bin" >&2
  exit 1
fi
profile_id="runpod_8xh100_parameter_golf"
trainer_lane_id="parameter_golf_distributed_8xh100"

run_root=""
submission_dir=""
output_path=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-finalize-8xh100.sh --run-root <path> --submission-dir <path> --output <path>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --submission-dir)
      submission_dir="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${run_root}" || -z "${submission_dir}" || -z "${output_path}" ]]; then
  echo "error: --run-root, --submission-dir, and --output are required" >&2
  usage
  exit 1
fi

resolve_submission_dir() {
  local requested_dir="$1"
  local run_root="$2"
  local candidate
  local -a candidates=(
    "${requested_dir}"
    "${run_root}/exported_submission"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/submission.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "error: could not resolve exported submission root from \`${requested_dir}\` or \`${run_root}/exported_submission\`" >&2
  exit 1
}

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

submission_dir="$(resolve_submission_dir "${submission_dir}" "${run_root}")"
run_id="$(basename -- "${run_root}")"
created_at_utc="$(timestamp_utc)"
distributed_receipt_path="${run_root}/parameter_golf_distributed_8xh100_receipt.json"
distributed_measurements_path="${run_root}/parameter_golf_distributed_8xh100_measurements.json"
visualization_bundle_path="${run_root}/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json"
visualization_run_index_path="${run_root}/training_visualization/remote_training_run_index_v1.json"
repo_revision="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "${repo_revision}" ]]; then
  repo_revision="workspace@${repo_revision}"
else
  repo_revision="workspace@unknown"
fi

mkdir -p "${run_root}" "$(dirname -- "${output_path}")"

inventory_file="${run_root}/nvidia_smi_inventory.txt"
topology_file="${run_root}/nvidia_smi_topology.txt"

nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "${inventory_file}"
nvidia-smi topo -m > "${topology_file}"

if [[ ! -f "${distributed_receipt_path}" && -f "${distributed_measurements_path}" ]]; then
  bash "${repo_root}/scripts/parameter-golf-runpod-build-8xh100-receipt.sh" \
    --run-root "${run_root}" \
    --measurements "${distributed_measurements_path}" \
    --output "${distributed_receipt_path}"
fi

execution_log_path="${run_root}/execution.log"
if [[ ! -f "${distributed_measurements_path}" && -f "${execution_log_path}" ]]; then
  bash "${repo_root}/scripts/parameter-golf-runpod-build-8xh100-measurements.sh" \
    --run-root "${run_root}" \
    --log "${execution_log_path}" \
    --output "${distributed_measurements_path}"
fi

if [[ ! -f "${distributed_receipt_path}" && -f "${distributed_measurements_path}" ]]; then
  bash "${repo_root}/scripts/parameter-golf-runpod-build-8xh100-receipt.sh" \
    --run-root "${run_root}" \
    --measurements "${distributed_measurements_path}" \
    --output "${distributed_receipt_path}"
fi

entrypoint_path="${submission_dir}/train_gpt.py"
manifest_path="${submission_dir}/submission.json"
run_evidence_path="${submission_dir}/psionic_parameter_golf_submission_run_evidence.json"

if [[ ! -f "${run_evidence_path}" ]]; then
  run_evidence_cmd=(
    cargo run -q -p psionic-train --example parameter_golf_submission_run_evidence
    --manifest-path "${repo_root}/crates/psionic-train/Cargo.toml"
    -- "${submission_dir}" "${run_evidence_path}" --posture runpod_8xh100
  )
  if [[ -f "${distributed_receipt_path}" ]]; then
    run_evidence_cmd+=(--distributed-receipt "${distributed_receipt_path}")
  fi
  "${run_evidence_cmd[@]}"
fi

python3 - "${run_evidence_path}" "${distributed_receipt_path}" <<'PY'
import json
import sys
from pathlib import Path

run_evidence_path = Path(sys.argv[1])
distributed_receipt_path = Path(sys.argv[2])

if distributed_receipt_path.exists():
    sys.exit(0)
if not run_evidence_path.is_file():
    sys.exit(0)

report = json.loads(run_evidence_path.read_text(encoding="utf-8"))
receipt = report.get("distributed_challenge_receipt")
if receipt is None:
    sys.exit(0)

distributed_receipt_path.write_text(
    json.dumps(receipt, indent=2) + "\n",
    encoding="utf-8",
)
PY

python3 - "${run_root}" "${submission_dir}" "${output_path}" "${inventory_file}" "${topology_file}" "${entrypoint_path}" "${manifest_path}" "${run_evidence_path}" "${distributed_receipt_path}" "${distributed_measurements_path}" "${created_at_utc}" "${run_id}" "${profile_id}" "${trainer_lane_id}" <<'PY'
import json
import sys
from hashlib import sha256 as sha256_hash
from pathlib import Path

run_root = Path(sys.argv[1])
submission_dir = Path(sys.argv[2])
output_path = Path(sys.argv[3])
inventory_file = Path(sys.argv[4])
topology_file = Path(sys.argv[5])
entrypoint_path = Path(sys.argv[6])
manifest_path = Path(sys.argv[7])
run_evidence_path = Path(sys.argv[8])
distributed_receipt_path = Path(sys.argv[9])
distributed_measurements_path = Path(sys.argv[10])
created_at_utc = sys.argv[11]
run_id = sys.argv[12]
profile_id = sys.argv[13]
trainer_lane_id = sys.argv[14]

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_hash(path.read_bytes()).hexdigest()

report = {
    "schema_version": "parameter_golf.runpod_8xh100_finalizer.v1",
    "runner": "scripts/parameter-golf-runpod-finalize-8xh100.sh",
    "created_at_utc": created_at_utc,
    "run_id": run_id,
    "profile_id": profile_id,
    "trainer_lane_id": trainer_lane_id,
    "run_root": str(run_root),
    "submission_dir": str(submission_dir),
    "world_size": 8,
    "grad_accum_steps": 1,
    "accelerator_evidence": {
      "inventory_path": str(inventory_file),
      "topology_path": str(topology_file),
      "inventory_line_count": len([line for line in inventory_file.read_text(encoding="utf-8").splitlines() if line.strip()]),
    },
    "exported_folder_evidence": {
      "entrypoint_path": str(entrypoint_path),
      "entrypoint_sha256": sha256(entrypoint_path),
      "submission_manifest_path": str(manifest_path),
      "submission_manifest_sha256": sha256(manifest_path),
      "submission_run_evidence_path": str(run_evidence_path) if run_evidence_path.exists() else None,
      "submission_run_evidence_sha256": sha256(run_evidence_path),
      "distributed_receipt_path": str(distributed_receipt_path) if distributed_receipt_path.exists() else None,
      "distributed_receipt_sha256": sha256(distributed_receipt_path),
      "distributed_measurements_path": str(distributed_measurements_path) if distributed_measurements_path.exists() else None,
      "distributed_measurements_sha256": sha256(distributed_measurements_path),
    },
    "claim_boundary": "This finalizer preserves the machine inventory, topology, exported-folder digests, and the RunPod 8xH100-bound submission run evidence surface. It does not by itself claim that the later real 8xH100 execution cleared the challenge bar."
}
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

(
  cd "${repo_root}"
  cargo run -q -p psionic-train --bin parameter_golf_distributed_visualization -- \
    --report "${output_path}" \
    --repo-revision "${repo_revision}"
)

python3 - "${output_path}" "${visualization_bundle_path}" "${visualization_run_index_path}" <<'PY'
import json
import sys
from hashlib import sha256 as sha256_hash
from pathlib import Path

report_path = Path(sys.argv[1])
bundle_path = Path(sys.argv[2])
run_index_path = Path(sys.argv[3])

report = json.loads(report_path.read_text(encoding="utf-8"))

def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_hash(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

bundle = load_json(bundle_path)
run_index = load_json(run_index_path)
report["remote_training_visualization"] = (
    {
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_digest": None if bundle is None else bundle.get("bundle_digest"),
        "run_index_path": str(run_index_path),
        "run_index_sha256": sha256_file(run_index_path),
        "run_index_digest": None if run_index is None else run_index.get("index_digest"),
        "result_classification": None if bundle is None else bundle.get("result_classification"),
        "series_status": None if bundle is None else bundle.get("series_status"),
        "last_heartbeat_at_ms": None
        if bundle is None
        else (bundle.get("refresh_contract") or {}).get("last_heartbeat_at_ms"),
        "heartbeat_seq": None
        if bundle is None
        else (bundle.get("refresh_contract") or {}).get("heartbeat_seq"),
    }
    if bundle_path.is_file() or run_index_path.is_file()
    else None
)
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
