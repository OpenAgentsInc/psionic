#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./TRAIN --lane cs336_a1_demo [start] [options]
  ./TRAIN --lane cs336_a1_demo rehearse-base-lane [options]
  ./TRAIN --lane cs336_a1_demo status --run-root <path>

Options for `start` and `rehearse-base-lane`:
  --run-id <id>            Stable run identifier.
  --output-root <path>     Local A1 demo run root. Default: ~/scratch/psion_cs336_a1_demo_runs/<run_id>
  --git-ref <ref>          Git ref to record for the packaged run. Default: HEAD
  --allow-dirty-tree       Override the default dirty-tree refusal and retain a status digest.
  --dry-run                Materialize the packaged retained surfaces without executing the four-step run.

Options for `status`:
  --run-root <path>        Existing A1 demo run root containing retained status, summary, and checkpoint surfaces.

This lane is the packaged bounded CS336 A1 demo path for Pylon/Nexus rehearsal.
It always uses the admitted tiny corpus and the fixed four-step training budget.
EOF
}

command="start"
if [[ $# -ge 1 ]]; then
  case "$1" in
    start|rehearse-base-lane|status)
      command="$1"
      shift
      ;;
    --help|-h|help)
      usage
      exit 0
      ;;
    *)
      ;;
  esac
fi

case "${command}" in
  start|rehearse-base-lane|status)
    exec cargo run -q -p psionic-train --bin psionic-train -- cs336-a1-demo "${command}" "$@"
    ;;
  *)
    echo "error: unsupported cs336_a1_demo command ${command}" >&2
    usage
    exit 1
    ;;
esac
