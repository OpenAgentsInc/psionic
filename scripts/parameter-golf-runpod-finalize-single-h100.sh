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

run_root=""
output_path=""
training_report_path=""
training_log_path=""
pod_id=""
pod_host=""
pod_port=""
ssh_user=""
trainer_pid=""
training_exit_code=""
failure_detail=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-finalize-single-h100.sh --run-root <path> --output <path> [options]

Options:
  --training-report <path>   Override the trainer JSON report path.
  --training-log <path>      Override the trainer log path.
  --pod-id <id>              Preserve the provider pod identifier.
  --pod-host <host>          Preserve the pod host or IP.
  --pod-port <port>          Preserve the exposed SSH port.
  --ssh-user <user>          Preserve the SSH username used for the pod.
  --trainer-pid <pid>        Inspect one live trainer process.
  --training-exit-code <n>   Preserve an explicit trainer exit code.
  --failure-detail <text>    Preserve an explicit failure or refusal detail.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --training-report)
      training_report_path="$2"
      shift 2
      ;;
    --training-log)
      training_log_path="$2"
      shift 2
      ;;
    --pod-id)
      pod_id="$2"
      shift 2
      ;;
    --pod-host)
      pod_host="$2"
      shift 2
      ;;
    --pod-port)
      pod_port="$2"
      shift 2
      ;;
    --ssh-user)
      ssh_user="$2"
      shift 2
      ;;
    --trainer-pid)
      trainer_pid="$2"
      shift 2
      ;;
    --training-exit-code)
      training_exit_code="$2"
      shift 2
      ;;
    --failure-detail)
      failure_detail="$2"
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

if [[ -z "${run_root}" || -z "${output_path}" ]]; then
  echo "error: --run-root and --output are required" >&2
  usage
  exit 1
fi

if [[ ! -d "${run_root}" ]]; then
  echo "error: run root does not exist: ${run_root}" >&2
  exit 1
fi

mkdir -p "${run_root}" "$(dirname -- "${output_path}")"

if [[ -z "${training_report_path}" ]]; then
  training_report_path="${run_root}/parameter_golf_single_h100_training.json"
fi
if [[ -z "${training_log_path}" ]]; then
  training_log_path="${run_root}/parameter_golf_single_h100_train.log"
fi

repo_revision="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "${repo_revision}" ]]; then
  repo_revision="workspace@unknown"
else
  repo_revision="workspace@${repo_revision}"
fi

inventory_file="${run_root}/nvidia_smi_inventory.txt"
topology_file="${run_root}/nvidia_smi_topology.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader > "${inventory_file}" || true
  nvidia-smi topo -m > "${topology_file}" || true
fi

python3 - "${run_root}" "${output_path}" "${training_report_path}" "${training_log_path}" "${inventory_file}" "${topology_file}" "${pod_id}" "${pod_host}" "${pod_port}" "${ssh_user}" "${trainer_pid}" "${training_exit_code}" "${failure_detail}" <<'PY'
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])
training_report_path = Path(sys.argv[3])
training_log_path = Path(sys.argv[4])
inventory_file = Path(sys.argv[5])
topology_file = Path(sys.argv[6])
pod_id = sys.argv[7]
pod_host = sys.argv[8]
pod_port = sys.argv[9]
ssh_user = sys.argv[10]
trainer_pid = sys.argv[11]
training_exit_code = sys.argv[12]
failure_detail = sys.argv[13]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_micro_step(lines: list[str]) -> dict | None:
    pattern = re.compile(
        r"micro_step_complete step=(?P<step>\d+)/(?P<max_steps>\d+) "
        r"micro_step=(?P<micro_step>\d+)/(?P<grad_accum_steps>\d+) .*?"
        r"train_loss=(?P<train_loss>[0-9.]+) "
        r"forward_ms=(?P<forward_ms>\d+) "
        r"backward_ms=(?P<backward_ms>\d+) "
        r"host_materialization_ms=(?P<host_materialization_ms>\d+)"
    )
    for line in reversed(lines):
        match = pattern.search(line)
        if not match:
            continue
        groups = match.groupdict()
        return {
            "global_step": int(groups["step"]),
            "max_steps": int(groups["max_steps"]),
            "micro_step": int(groups["micro_step"]),
            "grad_accum_steps": int(groups["grad_accum_steps"]),
            "train_loss": float(groups["train_loss"]),
            "forward_ms": int(groups["forward_ms"]),
            "backward_ms": int(groups["backward_ms"]),
            "host_materialization_ms": int(groups["host_materialization_ms"]),
        }
    return None


def parse_process(pid: str) -> dict | None:
    if not pid:
        return None
    try:
        output = subprocess.check_output(
            ["ps", "-p", pid, "-o", "pid=", "-o", "etime=", "-o", "stat=", "-o", "pcpu=", "-o", "pmem=", "-o", "cmd="],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return {
            "pid": int(pid),
            "running": False,
        }
    if not output:
        return {
            "pid": int(pid),
            "running": False,
        }
    parts = output.split(None, 5)
    return {
        "pid": int(parts[0]),
        "elapsed": parts[1],
        "state": parts[2],
        "cpu_percent": float(parts[3]),
        "memory_percent": float(parts[4]),
        "command": parts[5] if len(parts) > 5 else "",
        "running": True,
    }


training_report = read_json(training_report_path)
training_log_lines: list[str] = []
if training_log_path.is_file():
    training_log_lines = training_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
latest_micro_step = parse_micro_step(training_log_lines)
trainer_process = parse_process(trainer_pid)

report_summary = None
if training_report is not None:
    pre_export_final_validation = training_report.get("pre_export_final_validation") or {}
    final_validation = training_report.get("final_validation") or {}
    final_roundtrip_receipt = training_report.get("final_roundtrip_receipt") or {}
    report_summary = {
        "path": str(training_report_path),
        "exists": True,
        "sha256": sha256(training_report_path),
        "run_id": training_report.get("run_id"),
        "disposition": training_report.get("disposition"),
        "executed_steps": training_report.get("executed_steps"),
        "pre_export_final_val_loss": pre_export_final_validation.get("mean_loss"),
        "pre_export_final_val_bpb": pre_export_final_validation.get("bits_per_byte"),
        "final_val_loss": final_validation.get("mean_loss"),
        "final_val_bpb": final_validation.get("bits_per_byte"),
        "final_roundtrip_eval_ms": final_roundtrip_receipt.get("observed_eval_ms"),
        "final_roundtrip_metric_source": final_roundtrip_receipt.get("metric_source"),
        "compressed_model_bytes": training_report.get("compressed_model_bytes"),
        "compressed_model_artifact_ref": training_report.get("compressed_model_artifact_ref"),
        "compressed_model_artifact_digest": training_report.get("compressed_model_artifact_digest"),
        "machine_contract_satisfied": training_report.get("machine_contract_satisfied"),
        "challenge_kernel_blockers": training_report.get("challenge_kernel_blockers") or [],
        "observed_wallclock_ms": training_report.get("observed_wallclock_ms"),
        "report_digest": training_report.get("report_digest"),
        "claim_boundary": training_report.get("claim_boundary"),
        "summary": training_report.get("summary"),
    }
else:
    report_summary = {
        "path": str(training_report_path),
        "exists": False,
        "sha256": None,
    }

if training_report is not None and training_report.get("disposition") == "training_executed":
    final_validation = training_report.get("final_validation") or {}
    if final_validation.get("mean_loss") is not None and final_validation.get("bits_per_byte") is not None and training_report.get("compressed_model_bytes") is not None:
        audit_status = "succeeded"
    else:
        audit_status = "failed_incomplete_training_report"
elif training_report is not None:
    audit_status = "refused"
elif trainer_process and trainer_process.get("running"):
    audit_status = "in_progress"
elif training_exit_code and training_exit_code != "0":
    audit_status = "failed_training_exit_nonzero"
else:
    audit_status = "failed_missing_training_report"

derived_failure_detail = failure_detail
if not derived_failure_detail:
    if audit_status == "in_progress":
        derived_failure_detail = "single-H100 trainer is still running; final JSON receipt has not been written yet"
    elif audit_status == "failed_missing_training_report":
        derived_failure_detail = "trainer process is not running and no final training report was found under the run root"
    elif audit_status == "failed_training_exit_nonzero":
        derived_failure_detail = f"trainer exited with non-zero status {training_exit_code} before writing the final report"
    elif audit_status == "failed_incomplete_training_report":
        derived_failure_detail = "training report exists but does not yet carry final validation metrics and compressed-model bytes"
    elif audit_status == "refused":
        derived_failure_detail = (training_report or {}).get("summary") or "trainer reported an explicit refusal outcome"
    else:
        derived_failure_detail = ""

report = {
    "schema_version": "parameter_golf.runpod_single_h100_audit.v1",
    "runner": "scripts/parameter-golf-runpod-finalize-single-h100.sh",
    "collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_root": str(run_root),
    "pod_identity": {
        "pod_id": pod_id or None,
        "pod_host": pod_host or None,
        "pod_port": int(pod_port) if pod_port else None,
        "ssh_user": ssh_user or None,
        "hostname": socket.gethostname(),
    },
    "trainer_process": trainer_process,
    "accelerator_evidence": {
        "inventory_path": str(inventory_file) if inventory_file.is_file() else None,
        "inventory_sha256": sha256(inventory_file),
        "inventory_line_count": len([line for line in inventory_file.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]) if inventory_file.is_file() else 0,
        "topology_path": str(topology_file) if topology_file.is_file() else None,
        "topology_sha256": sha256(topology_file),
    },
    "training_report": report_summary,
    "training_log": {
        "path": str(training_log_path),
        "exists": training_log_path.is_file(),
        "sha256": sha256(training_log_path),
        "line_count": len(training_log_lines),
        "latest_micro_step": latest_micro_step,
        "tail": training_log_lines[-20:],
    },
    "training_exit_code": int(training_exit_code) if training_exit_code else None,
    "audit_status": audit_status,
    "failure_detail": derived_failure_detail or None,
    "claim_boundary": "This audit preserves the exact RunPod single-H100 run root, pod identity, accelerator evidence, trainer report metrics when present, and the latest retained log progress or failure cause. It does not by itself promote the lane beyond the exact preserved single-H100 outcome.",
}

output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

audit_status="$(
  python3 - "${output_path}" <<'PY'
import json
import sys

print(json.loads(open(sys.argv[1], encoding="utf-8").read())["audit_status"])
PY
)"

case "${audit_status}" in
  succeeded)
    visualization_result_classification="completed_success"
    ;;
  refused)
    visualization_result_classification="refused"
    ;;
  in_progress)
    visualization_result_classification="active"
    ;;
  *)
    visualization_result_classification="completed_failure"
    ;;
esac

(
  cd "${repo_root}"
  cargo run -q -p psionic-train --bin parameter_golf_single_h100_visualization -- \
    --run-root "${run_root}" \
    --training-report "${training_report_path}" \
    --training-log "${training_log_path}" \
    --provider run_pod \
    --profile-id runpod_h100_single_gpu \
    --lane-id parameter_golf.runpod_single_h100 \
    --repo-revision "${repo_revision}" \
    --result-classification "${visualization_result_classification}"
)

python3 - "${output_path}" "${run_root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
visualization_dir = run_root / "training_visualization"
bundle_path = visualization_dir / "parameter_golf_single_h100_remote_training_visualization_bundle_v1.json"
run_index_path = visualization_dir / "remote_training_run_index_v1.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


report = read_json(output_path) or {}
bundle = read_json(bundle_path) or {}
run_index = read_json(run_index_path) or {}
report["remote_training_visualization"] = {
    "bundle_path": str(bundle_path),
    "bundle_exists": bundle_path.is_file(),
    "bundle_sha256": sha256(bundle_path),
    "bundle_digest": bundle.get("bundle_digest"),
    "run_index_path": str(run_index_path),
    "run_index_exists": run_index_path.is_file(),
    "run_index_sha256": sha256(run_index_path),
    "run_index_digest": run_index.get("index_digest"),
    "provider": bundle.get("provider"),
    "profile_id": bundle.get("profile_id"),
    "lane_id": bundle.get("lane_id"),
    "result_classification": bundle.get("result_classification"),
    "series_status": bundle.get("series_status"),
    "last_heartbeat_at_ms": ((bundle.get("refresh_contract") or {}).get("last_heartbeat_at_ms")),
    "heartbeat_seq": ((bundle.get("refresh_contract") or {}).get("heartbeat_seq")),
}
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
