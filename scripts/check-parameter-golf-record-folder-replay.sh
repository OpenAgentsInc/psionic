#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
submission_dir=""
report_path=""

usage() {
    cat <<'EOF' >&2
Usage: scripts/check-parameter-golf-record-folder-replay.sh [--submission-dir <path>] [--report <path>]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submission-dir)
            [[ $# -ge 2 ]] || {
                echo "missing path after --submission-dir" >&2
                usage
                exit 1
            }
            submission_dir="$2"
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

if [[ -z "$submission_dir" ]]; then
    submission_dir="$(pwd)"
fi

cd "$repo_root"
if [[ -n "$report_path" ]]; then
    cargo run -q -p psionic-train --example parameter_golf_record_folder_replay_verification -- "$submission_dir" "$report_path"
else
    cargo run -q -p psionic-train --example parameter_golf_record_folder_replay_verification -- "$submission_dir"
fi
