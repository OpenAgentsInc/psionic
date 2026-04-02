#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF' >&2
Usage:
  ./TRAIN --lane actual_pretraining start [options]
  ./TRAIN --lane actual_pretraining record-checkpoint --run-root <path> --checkpoint-label <label> --optimizer-step <step> --checkpoint-ref <ref> [options]
  ./TRAIN --lane actual_pretraining backup --run-root <path> [options]
  ./TRAIN --lane actual_pretraining resume --run-root <path> [options]
  ./TRAIN --lane actual_pretraining status --run-root <path>

Options for `start`:
  --run-id <id>            Stable run identifier.
  --output-root <path>     Local actual-lane run root. Default: ~/scratch/psion_actual_pretraining_runs/<run_id>
  --git-ref <ref>          Git ref to resolve for the run. Default: current symbolic ref or HEAD
  --hardware-observation <path>
                           Optional retained hardware observation snapshot to consume instead of probing the local host.
  --run-shape-observation <path>
                           Optional retained throughput/storage/dataloader observation snapshot to consume instead of probing the local host.
  --allow-dirty-tree       Override the default dirty-tree refusal and retain a status digest.
  --dry-run                Materialize the retained launcher bundle without claiming cluster execution.

Options for `resume`:
  --run-root <path>        Existing actual-lane run root containing checkpoints/latest_accepted_checkpoint_pointer.json
  --git-ref <ref>          Git ref to resolve for the resumed run. Default: current symbolic ref or HEAD
  --hardware-observation <path>
                           Optional retained hardware observation snapshot to consume instead of probing the local host.
  --run-shape-observation <path>
                           Optional retained throughput/storage/dataloader observation snapshot to consume instead of probing the local host.
  --allow-dirty-tree       Override the default dirty-tree refusal and retain a status digest.
  --dry-run                Materialize the retained resume bundle without claiming cluster execution.

Options for `record-checkpoint`:
  --run-root <path>        Existing actual-lane run root to update with an accepted checkpoint.
  --checkpoint-label <label>
                           Stable accepted checkpoint label.
  --optimizer-step <step>  Accepted optimizer step.
  --checkpoint-ref <ref>   Stable checkpoint ref for later resume and continuation.
  --checkpoint-object-digest <digest>
                           Optional checkpoint object digest. Default: stable synthetic digest over the accepted checkpoint identity.
  --checkpoint-total-bytes <bytes>
                           Optional checkpoint byte size. Default: frozen actual-lane checkpoint size from the systems bundle.
  --git-ref <ref>          Git ref to resolve for the checkpoint provenance. Default: current symbolic ref or HEAD
  --allow-dirty-tree       Override the default dirty-tree refusal and retain a status digest.
  --inject-eval-worker-unavailable
                           Retain a checkpoint-eval retry receipt and redacted alert instead of a successful automatic checkpoint eval decision.

Options for `backup`:
  --run-root <path>        Existing actual-lane run root containing an accepted checkpoint pointer and manifest.
  --git-ref <ref>          Git ref to resolve for the backup provenance. Default: current symbolic ref or HEAD
  --allow-dirty-tree       Override the default dirty-tree refusal and retain a status digest.
  --inject-failed-upload   Retain a failed-upload drill receipt instead of a successful durable-backup receipt.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

command="$1"
shift

case "${command}" in
  start|record-checkpoint|backup|resume)
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
