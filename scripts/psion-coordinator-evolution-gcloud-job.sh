#!/usr/bin/env bash
#
# Ready-to-run GCloud training-job spec for the coordinator evolution lane
# (Khala M6, P3-P5: sep-CMA-ES + reward adapter + worker-pool binding).
#
# DESIGN INTENT (conservative on spend):
#   - Separable CMA-ES is GRADIENT-FREE and CPU-friendly. The TRINITY-scale
#     head is ~10K params, so the optimizer itself does NOT need a GPU. The
#     expensive part is the per-eval (P4), which on the live lane runs real
#     workers / moves sats -- that is metered and budgeted, NOT a VM cost.
#   - Therefore this job runs on a SMALL CPU VM by default (e2-standard-4) and
#     runs the bounded CPU smoke. It does NOT provision a GPU.
#   - It is DRY-RUN by default: it prints the exact `gcloud` commands and the
#     job spec, and provisions NOTHING. Pass --submit to actually create the VM.
#
# Project: openagentsgemini (authenticated as chris@openagents.com).
#
# Usage:
#   scripts/psion-coordinator-evolution-gcloud-job.sh            # dry-run (default)
#   scripts/psion-coordinator-evolution-gcloud-job.sh --submit   # create the CPU VM job
#   scripts/psion-coordinator-evolution-gcloud-job.sh --teardown # delete the VM
#
# A real (non-smoke) ES training run requires, ON TOP OF this scaffold:
#   - a live `CoordinatorFitness` impl wired to the rollout coordinator
#     (`probe_gepa_rollout_coordinator.rs`) instead of the fixture verdict;
#   - the live `forward_with_hidden` feature on a frozen backbone (Qwen3-0.6B);
#   - the Tassadar `training.verification_classes.v1` verdict as the reward;
#   - a per-generation eval budget cap (sats), emitted as a receipt.
# See docs/COORDINATOR_EVOLUTION_TRAINING.md.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
ZONE="${ZONE:-us-central1-a}"
# CPU-only: sep-CMA-ES does not need an accelerator at TRINITY head scale.
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
INSTANCE_NAME="${INSTANCE_NAME:-psion-coord-evo-smoke}"
IMAGE_FAMILY="${IMAGE_FAMILY:-debian-12}"
IMAGE_PROJECT="${IMAGE_PROJECT:-debian-cloud}"
# Bounded so an accidental submit cannot become an open-ended bill.
MAX_RUN_DURATION="${MAX_RUN_DURATION:-900s}"

MODE="dry-run"
for arg in "$@"; do
  case "$arg" in
    --submit) MODE="submit" ;;
    --teardown) MODE="teardown" ;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Startup script run on the VM: build the crate and run the CPU smoke binary,
# then self-delete so the VM never lingers (defense-in-depth on spend).
read -r -d '' STARTUP_SCRIPT <<'STARTUP' || true
#!/usr/bin/env bash
set -euxo pipefail
apt-get update -y
apt-get install -y git build-essential curl pkg-config libssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
git clone --depth 1 --branch agent/psion-train \
  https://github.com/OpenAgentsInc/psionic.git /tmp/psionic || \
  git clone --depth 1 https://github.com/OpenAgentsInc/psionic.git /tmp/psionic
cd /tmp/psionic
# CPU-only ES smoke. Real training swaps the fixture eval for the live hook.
cargo run --release -q -p psionic-train --bin coordinator_evolution_smoke \
  2>&1 | tee /tmp/coordinator_evolution_smoke.log
# Self-delete so the VM does not linger past the bounded job.
NAME="$(curl -s -H 'Metadata-Flavor: Google' http://metadata/computeMetadata/v1/instance/name)"
ZONE_PATH="$(curl -s -H 'Metadata-Flavor: Google' http://metadata/computeMetadata/v1/instance/zone)"
ZONE_NAME="${ZONE_PATH##*/}"
gcloud --quiet compute instances delete "$NAME" --zone "$ZONE_NAME" || true
STARTUP

create_cmd=(
  gcloud compute instances create "$INSTANCE_NAME"
  --project "$PROJECT_ID"
  --zone "$ZONE"
  --machine-type "$MACHINE_TYPE"
  --image-family "$IMAGE_FAMILY"
  --image-project "$IMAGE_PROJECT"
  --max-run-duration "$MAX_RUN_DURATION"
  --instance-termination-action DELETE
  --no-restart-on-failure
  --scopes "https://www.googleapis.com/auth/cloud-platform"
  --metadata startup-script="$STARTUP_SCRIPT"
)

case "$MODE" in
  dry-run)
    echo "== DRY RUN (no spend) -- coordinator evolution GCloud job =="
    echo "project        : $PROJECT_ID"
    echo "zone           : $ZONE"
    echo "machine type   : $MACHINE_TYPE  (CPU-only; sep-CMA-ES needs no GPU)"
    echo "instance       : $INSTANCE_NAME"
    echo "max run        : $MAX_RUN_DURATION  (bounded; instance auto-DELETEs)"
    echo
    echo "Would run:"
    printf '  %q' "${create_cmd[@]}"; echo
    echo
    echo "Pass --submit to actually create the VM. Pass --teardown to delete it."
    ;;
  submit)
    echo "== SUBMIT -- creating bounded CPU VM (auto-deletes on completion) =="
    "${create_cmd[@]}"
    echo "Submitted. Tail the smoke log via:"
    echo "  gcloud compute ssh $INSTANCE_NAME --zone $ZONE --project $PROJECT_ID \\"
    echo "    --command 'tail -f /tmp/coordinator_evolution_smoke.log'"
    ;;
  teardown)
    echo "== TEARDOWN -- deleting $INSTANCE_NAME =="
    gcloud --quiet compute instances delete "$INSTANCE_NAME" \
      --zone "$ZONE" --project "$PROJECT_ID" || true
    ;;
esac
