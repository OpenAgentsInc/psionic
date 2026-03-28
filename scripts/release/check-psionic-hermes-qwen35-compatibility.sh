#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

hermes_root="${PSIONIC_HERMES_ROOT:-}"
if [[ -z "$hermes_root" ]]; then
  echo "set PSIONIC_HERMES_ROOT to a local Hermes checkout" >&2
  exit 1
fi
if [[ ! -d "$hermes_root" ]]; then
  echo "missing Hermes checkout: $hermes_root" >&2
  exit 1
fi

python_bin="${PSIONIC_HERMES_PYTHON:-$hermes_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "missing Hermes Python interpreter: $python_bin" >&2
  echo "set PSIONIC_HERMES_PYTHON to override the default path" >&2
  exit 1
fi

server_bin="${PSIONIC_HERMES_SERVER_BIN:-$repo_root/target/debug/psionic-openai-server}"
if [[ ! -x "$server_bin" ]]; then
  cargo build -q -p psionic-serve --bin psionic-openai-server
fi

model_path="${PSIONIC_HERMES_QWEN35_MODEL_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf}"
if [[ ! -f "$model_path" ]]; then
  echo "missing qwen35 model artifact: $model_path" >&2
  echo "set PSIONIC_HERMES_QWEN35_MODEL_PATH to override the default path" >&2
  exit 1
fi

port="${PSIONIC_HERMES_QWEN35_PORT:-8095}"
backend="${PSIONIC_HERMES_BACKEND:-cuda}"
host="${PSIONIC_HERMES_HOST:-127.0.0.1}"
base_url="http://${host}:${port}/v1"
report_path="${PSIONIC_HERMES_REPORT_PATH:-$repo_root/fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report.json}"
log_path="${PSIONIC_HERMES_SERVER_LOG_PATH:-$repo_root/target/hermes/hermes_qwen35_psionic_server.log}"
tmpdir="${TMPDIR:-$repo_root/target/hermes/tmp}"
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

TMPDIR="$tmpdir" "$server_bin" \
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

OPENAI_API_KEY=dummy \
OPENAI_BASE_URL="$base_url" \
"$python_bin" "$repo_root/scripts/release/hermes_qwen35_compatibility_probe.py" \
  --hermes-root "$hermes_root" \
  --base-url "$base_url" \
  --model "$model_name" \
  --model-path "$model_path" \
  --psionic-root "$repo_root" \
  --report-path "$report_path" \
  --backend-label psionic
