#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
BUCKET_URL="${BUCKET_URL:-gs://openagentsgemini-psion-train-us-central1}"
RUN_ID="${RUN_ID:-}"
MANIFEST_URI="${MANIFEST_URI:-}"
FORCE=false

usage() {
  cat <<'EOF'
Usage: psion-google-delete-two-node-swarm.sh [options]

Options:
  --run-id <run_id>           Resolve the dual-node cluster manifest from the training bucket.
  --manifest-uri <uri>        Use one explicit cluster manifest object.
  --force                     Delete even when the expected final manifest is missing.
EOF
}

wait_for_object() {
  local object_path="$1"
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if gcloud storage ls "${object_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --manifest-uri)
      MANIFEST_URI="$2"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ID}" && -z "${MANIFEST_URI}" ]]; then
  echo "error: provide --run-id or --manifest-uri" >&2
  exit 1
fi

if [[ -z "${MANIFEST_URI}" ]]; then
  MANIFEST_URI="${BUCKET_URL}/runs/${RUN_ID}/launch/psion_google_two_node_swarm_cluster_manifest.json"
fi

if ! wait_for_object "${MANIFEST_URI}"; then
  echo "error: cluster manifest ${MANIFEST_URI} does not exist" >&2
  exit 1
fi

manifest_json="$(gcloud storage cat "${MANIFEST_URI}")"
final_manifest_uri="$(jq -r '.final_manifest_uri' <<<"${manifest_json}")"
if [[ "${FORCE}" != "true" ]] && ! wait_for_object "${final_manifest_uri}"; then
  echo "error: final manifest ${final_manifest_uri} is missing; rerun with --force to bypass retention enforcement" >&2
  exit 1
fi

while IFS= read -r node_row; do
  instance_name="$(jq -r '.instance_name' <<<"${node_row}")"
  zone="$(jq -r '.zone' <<<"${node_row}")"
  if gcloud compute instances describe "${instance_name}" --project="${PROJECT_ID}" --zone="${zone}" >/dev/null 2>&1; then
    gcloud compute instances delete "${instance_name}" \
      --project="${PROJECT_ID}" \
      --zone="${zone}" \
      --quiet >/dev/null
    echo "deleted instance ${instance_name} in ${zone}"
  else
    echo "instance ${instance_name} was already absent"
  fi
done < <(jq -c '.nodes[]' <<<"${manifest_json}")
