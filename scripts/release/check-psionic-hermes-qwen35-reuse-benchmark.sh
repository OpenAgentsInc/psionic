#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

server_bin="${PSIONIC_HERMES_SERVER_BIN:-$repo_root/target/debug/psionic-openai-server}"
if [[ "${PSIONIC_HERMES_SKIP_BUILD:-0}" != "1" ]]; then
  cargo build -q -p psionic-serve --bin psionic-openai-server
fi
if [[ ! -x "$server_bin" ]]; then
  echo "missing psionic-openai-server binary after build: $server_bin" >&2
  echo "set PSIONIC_HERMES_SERVER_BIN and CARGO_TARGET_DIR together if the binary lives outside the repo target dir" >&2
  exit 1
fi

model_path="${PSIONIC_HERMES_QWEN35_MODEL_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf}"
if [[ ! -f "$model_path" ]]; then
  echo "missing qwen35 model artifact: $model_path" >&2
  echo "set PSIONIC_HERMES_QWEN35_MODEL_PATH to override the default path" >&2
  exit 1
fi

port="${PSIONIC_HERMES_QWEN35_PORT:-8097}"
backend="${PSIONIC_HERMES_BACKEND:-cuda}"
host="${PSIONIC_HERMES_HOST:-127.0.0.1}"
base_url="http://${host}:${port}/v1"
report_path="${PSIONIC_HERMES_REUSE_REPORT_PATH:-$repo_root/fixtures/qwen35/hermes/hermes_qwen35_reuse_benchmark.json}"
log_path="${PSIONIC_HERMES_SERVER_LOG_PATH:-$repo_root/target/hermes/hermes_qwen35_reuse_server.log}"
tmpdir="${TMPDIR:-$repo_root/target/hermes/tmp}"
iterations="${PSIONIC_HERMES_REUSE_ITERATIONS:-3}"
psionic_revision="${PSIONIC_HERMES_PSIONIC_REVISION:-}"

mkdir -p "$(dirname "$report_path")" "$(dirname "$log_path")" "$tmpdir"

model_name="$(basename "$model_path")"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

TMPDIR="$tmpdir" \
PSIONIC_OPENAI_INCLUDE_DEBUG_FIELDS=1 \
"$server_bin" \
  --backend "$backend" \
  --host "$host" \
  --port "$port" \
  -m "$model_path" \
  >"$log_path" 2>&1 &
server_pid="$!"

for _ in $(seq 1 60); do
  if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
  echo "psionic-openai-server failed health check on ${host}:${port}" >&2
  exit 1
fi

python3 "$repo_root/scripts/release/hermes_qwen35_reuse_benchmark.py" \
  --base-url "$base_url" \
  --model "$model_name" \
  --model-path "$model_path" \
  --psionic-root "$repo_root" \
  ${psionic_revision:+--psionic-revision "$psionic_revision"} \
  --report-path "$report_path" \
  --iterations "$iterations"
