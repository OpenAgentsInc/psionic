#!/usr/bin/env bash

set -euo pipefail

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

mkdir -p "${run_root}" "$(dirname -- "${output_path}")"

inventory_file="${run_root}/nvidia_smi_inventory.txt"
topology_file="${run_root}/nvidia_smi_topology.txt"

nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "${inventory_file}"
nvidia-smi topo -m > "${topology_file}"

sha256_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    return 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

entrypoint_path="${submission_dir}/train_gpt.py"
manifest_path="${submission_dir}/submission.json"
run_evidence_path="${submission_dir}/psionic_parameter_golf_submission_run_evidence.json"

python3 - "${run_root}" "${submission_dir}" "${output_path}" "${inventory_file}" "${topology_file}" "${entrypoint_path}" "${manifest_path}" "${run_evidence_path}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
submission_dir = Path(sys.argv[2])
output_path = Path(sys.argv[3])
inventory_file = Path(sys.argv[4])
topology_file = Path(sys.argv[5])
entrypoint_path = Path(sys.argv[6])
manifest_path = Path(sys.argv[7])
run_evidence_path = Path(sys.argv[8])

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]
    except Exception:
        return None

report = {
    "schema_version": "parameter_golf.runpod_8xh100_finalizer.v1",
    "runner": "scripts/parameter-golf-runpod-finalize-8xh100.sh",
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
    },
    "claim_boundary": "This finalizer preserves the machine inventory, topology, and exported-folder digest surface for the RunPod 8xH100 lane. It does not by itself claim that the later real 8xH100 execution cleared the challenge bar."
}
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
