#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_path=""
run_root=""
run_id="psion-cs336-a1-demo-check"
mode="existing_run"

usage() {
  cat <<'EOF' >&2
Usage: check-psion-cs336-a1-demo-lane.sh [--run-root <path>] [--report <path>] [--run-id <id>]

Without --run-root, the checker launches one fresh bounded rehearsal run into a
temporary run root and then verifies the retained outputs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --report)
      report_path="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
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

if [[ -z "${report_path}" ]]; then
  report_path="$(mktemp "${TMPDIR:-/tmp}/psion_cs336_a1_demo_check.XXXXXX").json"
fi

if [[ -z "${run_root}" ]]; then
  mode="fresh_rehearsal"
  run_root="$(mktemp -d "${TMPDIR:-/tmp}/psion_cs336_a1_demo_run.XXXXXX")"
  bash "${repo_root}/scripts/train-psion-cs336-a1-demo.sh" \
    rehearse-base-lane \
    --run-id "${run_id}" \
    --output-root "${run_root}" \
    --git-ref HEAD >/dev/null
fi

verification_json="$(
  bash "${repo_root}/scripts/train-psion-cs336-a1-demo.sh" \
    verify \
    --run-root "${run_root}"
)"

jq -n \
  --arg schema_version "psion.cs336_a1_demo_lane_check.v1" \
  --arg runner "scripts/check-psion-cs336-a1-demo-lane.sh" \
  --arg mode "${mode}" \
  --arg run_root "${run_root}" \
  --arg report_path "${report_path}" \
  --argjson verification "${verification_json}" \
  '{
    schema_version: $schema_version,
    runner: $runner,
    mode: $mode,
    run_root: $run_root,
    report_path: $report_path,
    verification: $verification
  }' > "${report_path}"

cat "${report_path}"
