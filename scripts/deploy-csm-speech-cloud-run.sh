#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROJECT_ID="${PSIONIC_GCP_PROJECT_ID:-${LYRA_GCP_PROJECT_ID:-openagents-lyra}}"
REGION="${PSIONIC_GCP_REGION:-us-central1}"
SERVICE="${PSIONIC_CSM_CLOUD_RUN_SERVICE:-psionic-csm-speech}"
REPOSITORY="${PSIONIC_ARTIFACT_REPOSITORY:-lyra}"
SERVICE_ACCOUNT_NAME="${PSIONIC_CSM_SERVICE_ACCOUNT:-psionic-csm-speech}"
ARTIFACT_BUCKET="${PSIONIC_CSM_ARTIFACT_BUCKET:-${PROJECT_ID}-psionic-csm-artifacts}"
BUILD_TIMEOUT="${PSIONIC_CSM_BUILD_TIMEOUT:-3600}"
BUILD_MACHINE_TYPE="${PSIONIC_CSM_BUILD_MACHINE_TYPE:-e2-highcpu-32}"
BUILD_DISK_SIZE="${PSIONIC_CSM_BUILD_DISK_SIZE:-200}"
CPU="${PSIONIC_CSM_CLOUD_RUN_CPU:-8}"
MEMORY="${PSIONIC_CSM_CLOUD_RUN_MEMORY:-32Gi}"
MAX_INSTANCES="${PSIONIC_CSM_CLOUD_RUN_MAX_INSTANCES:-1}"
MIN_INSTANCES="${PSIONIC_CSM_CLOUD_RUN_MIN_INSTANCES:-1}"
CONCURRENCY="${PSIONIC_CSM_CLOUD_RUN_CONCURRENCY:-1}"
REQUEST_TIMEOUT="${PSIONIC_CSM_CLOUD_RUN_TIMEOUT:-300}"
PORT="${PSIONIC_CSM_PORT:-8081}"
MODEL_ID="${PSIONIC_CSM_MODEL_ID:-sesame/csm-1b}"
BACKEND="${PSIONIC_CSM_BACKEND:-cpu}"
TAG="${PSIONIC_CSM_IMAGE_TAG:-$(git rev-parse --short HEAD)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE}:${TAG}"
SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

CSM_CONFIG_SHA="b203c014cb5a2f7b4f98d2e945f091182aceb17fa530ce968e8c3437e01a9b70"
CSM_MODEL_SHA="2e7721144afe38b906d4f1048671da639fe142423f4a26283606ecebe894f4bf"
LLAMA_TOKENIZER_SHA="79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"
MIMI_WEIGHT_SHA="09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
MIMI_WEIGHT_NAME="tokenizer-e351c8d8-checkpoint125.safetensors"

if [[ -n "${PSIONIC_CSM_HF_CACHE_ROOT:-}" ]]; then
  HF_CACHE_ROOT="$PSIONIC_CSM_HF_CACHE_ROOT"
elif [[ -n "${HF_HOME:-}" ]]; then
  HF_CACHE_ROOT="${HF_HOME}/hub"
else
  HF_CACHE_ROOT="${HOME}/.cache/huggingface/hub"
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

sha256_hex() {
  shasum -a 256 "$1" | awk '{print $1}'
}

find_hf_blob_or_matching_file() {
  local repo="$1"
  local expected_sha="$2"
  local file_name="$3"
  local blob="${HF_CACHE_ROOT}/${repo}/blobs/${expected_sha}"

  if [[ -f "$blob" ]]; then
    printf "%s" "$blob"
    return 0
  fi

  local snapshots="${HF_CACHE_ROOT}/${repo}/snapshots"
  if [[ ! -d "$snapshots" ]]; then
    return 1
  fi

  while IFS= read -r -d '' candidate; do
    if [[ "$(sha256_hex "$candidate")" == "$expected_sha" ]]; then
      printf "%s" "$candidate"
      return 0
    fi
  done < <(find "$snapshots" -type f -name "$file_name" -print0)

  return 1
}

find_hf_snapshot_file() {
  local repo="$1"
  local expected_sha="$2"
  local file_name="$3"
  local snapshots="${HF_CACHE_ROOT}/${repo}/snapshots"

  if [[ ! -d "$snapshots" ]]; then
    return 1
  fi

  while IFS= read -r -d '' candidate; do
    if [[ "$(sha256_hex "$candidate")" == "$expected_sha" ]]; then
      local snapshot
      snapshot="$(basename "$(dirname "$candidate")")"
      printf "%s\t%s" "$candidate" "$snapshot"
      return 0
    fi
  done < <(find "$snapshots" -type f -name "$file_name" -print0)

  return 1
}

copy_artifact_if_missing() {
  local source_path="$1"
  local object_path="$2"
  local uri="gs://${ARTIFACT_BUCKET}/${object_path}"

  if gcloud storage ls "$uri" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "artifact already staged: ${uri}"
    return 0
  fi

  echo "staging artifact: ${uri}"
  gcloud storage cp "$source_path" "$uri" --project "$PROJECT_ID"
}

ensure_artifact_repo() {
  if gcloud artifacts repositories describe "$REPOSITORY" \
    --project "$PROJECT_ID" \
    --location "$REGION" >/dev/null 2>&1; then
    return 0
  fi

  gcloud artifacts repositories create "$REPOSITORY" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --repository-format docker \
    --description "OpenAgents speech service container images"
}

wait_for_service_account() {
  local attempt
  for attempt in {1..24}; do
    if gcloud iam service-accounts describe "$SA_EMAIL" \
      --project "$PROJECT_ID" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done

  echo "service account did not become visible: ${SA_EMAIL}" >&2
  exit 1
}

ensure_service_account() {
  if gcloud iam service-accounts describe "$SA_EMAIL" \
    --project "$PROJECT_ID" >/dev/null 2>&1; then
    wait_for_service_account
    return 0
  fi

  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --project "$PROJECT_ID" \
    --display-name "Psionic CSM speech Cloud Run service"
  wait_for_service_account
}

ensure_artifact_bucket() {
  gcloud services enable storage.googleapis.com --project "$PROJECT_ID" >/dev/null

  if ! gcloud storage buckets describe "gs://${ARTIFACT_BUCKET}" \
    --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://${ARTIFACT_BUCKET}" \
      --project "$PROJECT_ID" \
      --location "$REGION" \
      --uniform-bucket-level-access \
      --public-access-prevention
  fi

  gcloud storage buckets update "gs://${ARTIFACT_BUCKET}" \
    --project "$PROJECT_ID" \
    --uniform-bucket-level-access \
    --public-access-prevention >/dev/null
}

grant_artifact_bucket_access() {
  gcloud storage buckets add-iam-policy-binding "gs://${ARTIFACT_BUCKET}" \
    --project "$PROJECT_ID" \
    --member "serviceAccount:${SA_EMAIL}" \
    --role roles/storage.objectViewer >/dev/null
}

stage_hf_artifacts() {
  if [[ ! -d "$HF_CACHE_ROOT" ]]; then
    echo "missing local Hugging Face cache root: ${HF_CACHE_ROOT}" >&2
    exit 1
  fi

  local csm_model_path
  csm_model_path="$(find_hf_blob_or_matching_file "models--sesame--csm-1b" "$CSM_MODEL_SHA" "model.safetensors")" || {
    echo "missing sesame/csm-1b model artifact with sha256:${CSM_MODEL_SHA}" >&2
    exit 1
  }

  local csm_config_path
  csm_config_path="$(find_hf_blob_or_matching_file "models--sesame--csm-1b" "$CSM_CONFIG_SHA" "config.json")" || {
    echo "missing sesame/csm-1b config artifact with sha256:${CSM_CONFIG_SHA}" >&2
    exit 1
  }

  local mimi_path
  mimi_path="$(find_hf_blob_or_matching_file "models--kyutai--moshiko-pytorch-bf16" "$MIMI_WEIGHT_SHA" "$MIMI_WEIGHT_NAME")" || {
    echo "missing kyutai/moshiko-pytorch-bf16 Mimi artifact with sha256:${MIMI_WEIGHT_SHA}" >&2
    exit 1
  }

  local tokenizer_record
  tokenizer_record="$(find_hf_snapshot_file "models--meta-llama--Llama-3.2-1B" "$LLAMA_TOKENIZER_SHA" "tokenizer.json")" || {
    echo "missing meta-llama/Llama-3.2-1B tokenizer.json with sha256:${LLAMA_TOKENIZER_SHA}" >&2
    exit 1
  }
  local tokenizer_path="${tokenizer_record%%$'\t'*}"
  local tokenizer_snapshot="${tokenizer_record##*$'\t'}"

  copy_artifact_if_missing "$csm_model_path" "hub/models--sesame--csm-1b/blobs/${CSM_MODEL_SHA}"
  copy_artifact_if_missing "$csm_config_path" "hub/models--sesame--csm-1b/blobs/${CSM_CONFIG_SHA}"
  copy_artifact_if_missing "$mimi_path" "hub/models--kyutai--moshiko-pytorch-bf16/blobs/${MIMI_WEIGHT_SHA}"
  copy_artifact_if_missing "$tokenizer_path" "hub/models--meta-llama--Llama-3.2-1B/snapshots/${tokenizer_snapshot}/tokenizer.json"
}

build_image() {
  local tmp_context
  tmp_context="$(mktemp -d)"
  trap 'rm -rf "$tmp_context"' RETURN

  git archive HEAD | tar -x -C "$tmp_context"
  cat >"${tmp_context}/Dockerfile" <<'DOCKERFILE'
FROM rust:1-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p psionic-serve --bin psionic-csm-speech-server

FROM debian:bookworm-slim
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/psionic-csm-speech-server /usr/local/bin/psionic-csm-speech-server
ENV PSIONIC_CSM_HOST=0.0.0.0
ENV PSIONIC_CSM_PORT=8081
ENV PSIONIC_CSM_RUNTIME=true
ENV PSIONIC_CSM_BACKEND=cpu
ENV HF_HOME=/root/.cache/huggingface
EXPOSE 8081
ENTRYPOINT ["/usr/local/bin/psionic-csm-speech-server"]
DOCKERFILE

  gcloud builds submit "$tmp_context" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --tag "$IMAGE" \
    --machine-type "$BUILD_MACHINE_TYPE" \
    --disk-size "$BUILD_DISK_SIZE" \
    --timeout "$BUILD_TIMEOUT"
}

deploy_service() {
  gcloud run deploy "$SERVICE" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --platform managed \
    --image "$IMAGE" \
    --service-account "$SA_EMAIL" \
    --execution-environment gen2 \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --timeout "$REQUEST_TIMEOUT" \
    --min-instances "$MIN_INSTANCES" \
    --max-instances "$MAX_INSTANCES" \
    --concurrency "$CONCURRENCY" \
    --no-cpu-throttling \
    --port "$PORT" \
    --allow-unauthenticated \
    --clear-volumes \
    --clear-volume-mounts \
    --add-volume "name=hf-cache,type=cloud-storage,bucket=${ARTIFACT_BUCKET},readonly=true,mount-options=implicit-dirs" \
    --add-volume-mount "volume=hf-cache,mount-path=/root/.cache/huggingface" \
    --startup-probe "tcpSocket.port=${PORT},periodSeconds=10,timeoutSeconds=5,failureThreshold=120" \
    --set-env-vars "PSIONIC_CSM_HOST=0.0.0.0,PSIONIC_CSM_PORT=${PORT},PSIONIC_CSM_MODEL_ID=${MODEL_ID},PSIONIC_CSM_RUNTIME=true,PSIONIC_CSM_BACKEND=${BACKEND},HF_HOME=/root/.cache/huggingface"
}

service_url() {
  gcloud run services describe "$SERVICE" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --format 'value(status.url)'
}

wait_for_ready_runtime() {
  local url="$1"
  local health_file
  health_file="$(mktemp)"
  trap 'rm -f "$health_file"' RETURN

  local attempt
  for attempt in {1..120}; do
    if curl -fsS --max-time 10 "${url}/health" -o "$health_file" >/dev/null 2>&1; then
      local runtime_state
      runtime_state="$(jq -r '.runtime.state // empty' "$health_file")"
      if [[ "$runtime_state" == "ready" ]]; then
        jq '{status,model,served_backend,execution_engine,runtime:.runtime}' "$health_file"
        return 0
      fi
      echo "runtime not ready yet: ${runtime_state:-unknown}"
    else
      echo "waiting for service health endpoint"
    fi
    sleep 10
  done

  echo "Psionic CSM service did not report runtime.state=ready: ${url}" >&2
  exit 1
}

smoke_speech() {
  local url="$1"
  local wav_file
  local headers_file
  wav_file="$(mktemp -t psionic-csm-prod.XXXXXX.wav)"
  headers_file="$(mktemp -t psionic-csm-prod.XXXXXX.headers)"
  trap 'rm -f "$wav_file" "$headers_file"' RETURN

  curl -fsS \
    -D "$headers_file" \
    -o "$wav_file" \
    -X POST "${url}/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d '{"model":"sesame/csm-1b","input":"hello from psionic production","voice_profile_id":"lyra/default_female_v1","response_format":"wav","psionic_csm":{"max_audio_length_ms":160,"context_policy":"none"}}'

  local bytes
  bytes="$(wc -c <"$wav_file" | tr -d ' ')"
  if [[ "$bytes" -le 44 ]]; then
    echo "speech smoke returned an invalid WAV payload: ${bytes} bytes" >&2
    exit 1
  fi

  echo "speech smoke ok: ${bytes} wav bytes"
  grep -i '^x-psionic-' "$headers_file" || true
}

require_command awk
require_command curl
require_command gcloud
require_command git
require_command jq
require_command shasum
require_command tar

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$PROJECT_ID" >/dev/null

ensure_artifact_repo
ensure_service_account
ensure_artifact_bucket
grant_artifact_bucket_access
stage_hf_artifacts
build_image
deploy_service

URL="$(service_url)"
wait_for_ready_runtime "$URL"
smoke_speech "$URL"

echo "$URL"
