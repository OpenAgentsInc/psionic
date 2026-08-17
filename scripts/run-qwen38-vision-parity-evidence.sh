#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${PSIONIC_QWEN38_OFFICIAL_MODEL_DIR:-$ROOT/target/models/qwen/Qwen3.8-27B}"
REFERENCE_PYTHON="${PSIONIC_QWEN38_TRANSFORMERS_PYTHON:?set PSIONIC_QWEN38_TRANSFORMERS_PYTHON}"
TRANSFORMERS_CHECKOUT="${PSIONIC_QWEN38_TRANSFORMERS_CHECKOUT:?set PSIONIC_QWEN38_TRANSFORMERS_CHECKOUT}"
IMAGE_REPORT_PATH="${PSIONIC_QWEN38_VISION_REPORT_PATH:-$ROOT/fixtures/qwen38/reports/qwen38_vision_parity_v1.json}"
VIDEO_REPORT_PATH="${PSIONIC_QWEN38_VISION_VIDEO_REPORT_PATH:-$ROOT/fixtures/qwen38/reports/qwen38_vision_video_parity_v1.json}"
SHARD="$MODEL_DIR/model-00001-of-00018.safetensors"
EXPECTED_SHARD_BYTES=3966730552
EXPECTED_SHARD_SHA256=ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c
EXPECTED_TRANSFORMERS_REVISION=0650ff354501cbdb7cb4138da628cc60f4e0ceed
IDLE_QUERY=(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "Qwen3.8 vision evidence requires branch main" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Qwen3.8 vision evidence requires a clean checkout" >&2
  exit 1
fi
git fetch origin main --quiet
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "Qwen3.8 vision evidence requires HEAD equal to origin/main" >&2
  exit 1
fi
if [[ "$(stat -c %s "$SHARD")" != "$EXPECTED_SHARD_BYTES" ]]; then
  echo "Qwen3.8 vision source shard byte length mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "$SHARD" | cut -d' ' -f1)" != "$EXPECTED_SHARD_SHA256" ]]; then
  echo "Qwen3.8 vision source shard SHA-256 mismatch" >&2
  exit 1
fi
if [[ "$(git -C "$TRANSFORMERS_CHECKOUT" rev-parse HEAD)" != "$EXPECTED_TRANSFORMERS_REVISION" ]]; then
  echo "Qwen3.8 Transformers comparator revision mismatch" >&2
  exit 1
fi

cargo test -p psionic-models qwen38_vision --lib
cargo test -p psionic-models qwen38_native_vision --lib
cargo build --release -p psionic-models --features qwen38-vision-cuda --example qwen38_vision_probe

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
run_probe() {
  local media="$1"
  local report_path="$2"
  local reference_json="$TMP_DIR/reference-$media.json"
  local native_json="$TMP_DIR/native-$media.json"

  if [[ -n "$("${IDLE_QUERY[@]}")" ]]; then
    echo "GPU has a resident compute process before the Transformers $media probe" >&2
    exit 1
  fi
  PYTHONPATH="$TRANSFORMERS_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$REFERENCE_PYTHON" scripts/qwen38-vision-transformers-reference.py \
    "$MODEL_DIR" "$TRANSFORMERS_CHECKOUT" "$media" > "$reference_json"

  if [[ -n "$("${IDLE_QUERY[@]}")" ]]; then
    echo "GPU has a resident compute process before the Psionic $media probe" >&2
    exit 1
  fi
  target/release/examples/qwen38_vision_probe "$MODEL_DIR" cuda "$media" > "$native_json"

  mkdir -p "$(dirname "$report_path")"
  "$REFERENCE_PYTHON" scripts/qwen38-vision-compare.py \
    "$native_json" \
    "$reference_json" \
    --psionic-revision "$(git rev-parse HEAD)" \
    --gpu-name "$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)" \
    --driver-version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)" \
    > "$report_path"

  bash scripts/check-qwen38-vision-parity.sh "$report_path"
  echo "$report_path"
}

run_probe image "$IMAGE_REPORT_PATH"
run_probe video "$VIDEO_REPORT_PATH"
