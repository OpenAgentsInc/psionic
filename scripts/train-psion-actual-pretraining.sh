#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF' >&2
Usage:
  ./TRAIN --lane actual_pretraining start [options]
  ./TRAIN --lane actual_pretraining resume --run-root <path> [options]
  ./TRAIN --lane actual_pretraining status --run-root <path>

Options for `start`:
  --run-id <id>            Stable run identifier.
  --output-root <path>     Local actual-lane run root. Default: ~/scratch/psion_actual_pretraining_runs/<run_id>
  --git-ref <ref>          Git ref to resolve for the run. Default: current symbolic ref or HEAD
  --allow-dirty-tree       Override the default dirty-tree refusal and retain a status digest.
  --dry-run                Materialize the retained launcher bundle without claiming cluster execution.

Options for `resume`:
  --run-root <path>        Existing actual-lane run root containing checkpoints/latest_accepted_checkpoint_pointer.json
  --git-ref <ref>          Git ref to resolve for the resumed run. Default: current symbolic ref or HEAD
  --allow-dirty-tree       Override the default dirty-tree refusal and retain a status digest.
  --dry-run                Materialize the retained resume bundle without claiming cluster execution.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

command="$1"
shift

case "${command}" in
  start|resume)
    exec cargo run -q -p psionic-train --example psion_actual_pretraining_operator -- "${command}" "$@"
    ;;
  status)
    exec "${script_dir}/psion-actual-pretraining-status.sh" "$@"
    ;;
  --help|-h|help)
    usage
    exit 0
    ;;
  *)
    echo "error: unsupported actual-pretraining command ${command}" >&2
    usage
    exit 1
    ;;
esac
