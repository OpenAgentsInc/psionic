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

psionic_model_path="${PSIONIC_HERMES_PSIONIC_MODEL_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf}"
if [[ ! -f "$psionic_model_path" ]]; then
  echo "missing Psionic qwen35 model artifact: $psionic_model_path" >&2
  echo "set PSIONIC_HERMES_PSIONIC_MODEL_PATH to override the default path" >&2
  exit 1
fi

ollama_root="${PSIONIC_HERMES_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
ollama_root="${ollama_root%/}"
ollama_model="${PSIONIC_HERMES_OLLAMA_MODEL:-qwen3.5:2b}"

psionic_host="${PSIONIC_HERMES_HOST:-127.0.0.1}"
psionic_port="${PSIONIC_HERMES_QWEN35_PORT:-8097}"
psionic_base_url="http://${psionic_host}:${psionic_port}/v1"
ollama_base_url="${ollama_root}/v1"

report_path="${PSIONIC_HERMES_BENCHMARK_REPORT_PATH:-$repo_root/fixtures/qwen35/hermes/hermes_psionic_vs_ollama_benchmark_report.json}"
raw_dir="${PSIONIC_HERMES_BENCHMARK_RAW_DIR:-$repo_root/fixtures/qwen35/hermes/backend_rows}"
psionic_row_path="${raw_dir}/psionic_row.json"
ollama_row_path="${raw_dir}/ollama_row.json"
psionic_log_path="${PSIONIC_HERMES_SERVER_LOG_PATH:-$repo_root/target/hermes/hermes_qwen35_psionic_benchmark_server.log}"
tmpdir="${TMPDIR:-$repo_root/target/hermes/tmp}"

mkdir -p "$(dirname "$report_path")" "$raw_dir" "$(dirname "$psionic_log_path")" "$tmpdir"

server_pid=""
llamacpp_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ -n "$llamacpp_pid" ]]; then
    kill "$llamacpp_pid" 2>/dev/null || true
    wait "$llamacpp_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

TMPDIR="$tmpdir" "$server_bin" \
  --backend cuda \
  --host "$psionic_host" \
  --port "$psionic_port" \
  -m "$psionic_model_path" \
  >"$psionic_log_path" 2>&1 &
server_pid="$!"

psionic_ready_start="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"
for _ in $(seq 1 60); do
  if curl -fsS "http://${psionic_host}:${psionic_port}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://${psionic_host}:${psionic_port}/health" >/dev/null 2>&1; then
  echo "psionic-openai-server failed health check on ${psionic_host}:${psionic_port}" >&2
  exit 1
fi
psionic_ready_end="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"

ollama_ready_start="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"
if ! curl -fsS "${ollama_root}/api/tags" >/dev/null 2>&1; then
  echo "ollama did not answer on ${ollama_root}" >&2
  exit 1
fi
ollama_ready_end="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"

enable_llamacpp="${PSIONIC_HERMES_ENABLE_LLAMACPP:-0}"
llamacpp_root_bin=""
llamacpp_port=""
llamacpp_base_url=""
llamacpp_row_path=""
llamacpp_ready_start=""
llamacpp_ready_end=""
llamacpp_model_alias=""
if [[ "$enable_llamacpp" == "1" ]]; then
  case "$(uname -s)" in
    Darwin)
      llamacpp_root_bin_default="/Users/christopherdavid/code/llama.cpp/build/bin/llama-server"
      llamacpp_ngl_default="4"
      ;;
    *)
      llamacpp_root_bin_default="/home/christopherdavid/code/llama.cpp/build/bin/llama-server"
      llamacpp_ngl_default="999"
      ;;
  esac
  llamacpp_root_bin="${PSIONIC_HERMES_LLAMA_CPP_BIN:-$llamacpp_root_bin_default}"
  if [[ ! -x "$llamacpp_root_bin" ]]; then
    echo "missing llama.cpp server binary: $llamacpp_root_bin" >&2
    exit 1
  fi
  llamacpp_model_path="${PSIONIC_HERMES_LLAMA_CPP_MODEL_PATH:-$psionic_model_path}"
  if [[ ! -f "$llamacpp_model_path" ]]; then
    echo "missing llama.cpp model artifact: $llamacpp_model_path" >&2
    exit 1
  fi
  llamacpp_port="${PSIONIC_HERMES_LLAMA_CPP_PORT:-8098}"
  llamacpp_host="${PSIONIC_HERMES_LLAMA_CPP_HOST:-127.0.0.1}"
  llamacpp_ctx="${PSIONIC_HERMES_LLAMA_CPP_CTX:-4096}"
  llamacpp_ngl="${PSIONIC_HERMES_LLAMA_CPP_NGL:-$llamacpp_ngl_default}"
  llamacpp_base_url="http://${llamacpp_host}:${llamacpp_port}/v1"
  llamacpp_row_path="${raw_dir}/llama_cpp_row.json"
  llamacpp_model_alias="${PSIONIC_HERMES_LLAMA_CPP_MODEL_ALIAS:-$(basename "$llamacpp_model_path")}"
  llamacpp_log_path="${PSIONIC_HERMES_LLAMA_CPP_LOG_PATH:-$repo_root/target/hermes/hermes_llamacpp_benchmark_server.log}"
  mkdir -p "$(dirname "$llamacpp_log_path")"
  "$llamacpp_root_bin" \
    -m "$llamacpp_model_path" \
    --host "$llamacpp_host" \
    --port "$llamacpp_port" \
    -c "$llamacpp_ctx" \
    -ngl "$llamacpp_ngl" \
    --alias "$llamacpp_model_alias" \
    >"$llamacpp_log_path" 2>&1 &
  llamacpp_pid="$!"
  llamacpp_ready_start="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"
  for _ in $(seq 1 60); do
    if curl -fsS "http://${llamacpp_host}:${llamacpp_port}/health" >/dev/null 2>&1 || curl -fsS "http://${llamacpp_host}:${llamacpp_port}/v1/models" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! curl -fsS "http://${llamacpp_host}:${llamacpp_port}/health" >/dev/null 2>&1 && ! curl -fsS "http://${llamacpp_host}:${llamacpp_port}/v1/models" >/dev/null 2>&1; then
    echo "llama.cpp server failed readiness check on ${llamacpp_host}:${llamacpp_port}" >&2
    exit 1
  fi
  llamacpp_ready_end="$(python3 - <<'PY'
import time
print(f"{time.monotonic():.9f}")
PY
)"
fi

psionic_model_name="$(basename "$psionic_model_path")"

OPENAI_API_KEY=dummy \
OPENAI_BASE_URL="$psionic_base_url" \
"$python_bin" "$repo_root/scripts/release/hermes_backend_benchmark_probe.py" \
  --hermes-root "$hermes_root" \
  --base-url "$psionic_base_url" \
  --model "$psionic_model_name" \
  --model-path "$psionic_model_path" \
  --psionic-root "$repo_root" \
  --report-path "$psionic_row_path" \
  --backend-label psionic || true

OPENAI_API_KEY=dummy \
OPENAI_BASE_URL="$ollama_base_url" \
"$python_bin" "$repo_root/scripts/release/hermes_backend_benchmark_probe.py" \
  --hermes-root "$hermes_root" \
  --base-url "$ollama_base_url" \
  --model "$ollama_model" \
  --model-path "$ollama_model" \
  --psionic-root "$repo_root" \
  --report-path "$ollama_row_path" \
  --backend-label ollama || true

if [[ "$enable_llamacpp" == "1" ]]; then
  OPENAI_API_KEY=dummy \
  OPENAI_BASE_URL="$llamacpp_base_url" \
  "$python_bin" "$repo_root/scripts/release/hermes_backend_benchmark_probe.py" \
    --hermes-root "$hermes_root" \
    --base-url "$llamacpp_base_url" \
    --model "$llamacpp_model_alias" \
    --model-path "$llamacpp_model_path" \
    --psionic-root "$repo_root" \
    --report-path "$llamacpp_row_path" \
    --backend-label llama_cpp || true
fi

python3 - "$report_path" "$psionic_row_path" "$ollama_row_path" "$psionic_ready_start" "$psionic_ready_end" "$ollama_ready_start" "$ollama_ready_end" "$enable_llamacpp" "$llamacpp_row_path" "$llamacpp_ready_start" "$llamacpp_ready_end" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

report_path = Path(sys.argv[1])
psionic_row_path = Path(sys.argv[2])
ollama_row_path = Path(sys.argv[3])
psionic_ready_start = float(sys.argv[4])
psionic_ready_end = float(sys.argv[5])
ollama_ready_start = float(sys.argv[6])
ollama_ready_end = float(sys.argv[7])
enable_llamacpp = sys.argv[8] == "1"
llamacpp_row_path = Path(sys.argv[9]) if enable_llamacpp else None
llamacpp_ready_start = float(sys.argv[10]) if enable_llamacpp else None
llamacpp_ready_end = float(sys.argv[11]) if enable_llamacpp else None

psionic_row = json.loads(psionic_row_path.read_text())
ollama_row = json.loads(ollama_row_path.read_text())

rows = [psionic_row, ollama_row]
if enable_llamacpp:
    rows.append(json.loads(llamacpp_row_path.read_text()))
by_label = {row["backend_label"]: row for row in rows}
case_ids = [case["case_id"] for case in psionic_row["cases"]]
comparison = []
for case_id in case_ids:
    case_map = {}
    for row in rows:
        row_case = next(case for case in row["cases"] if case["case_id"] == case_id)
        case_map[row["backend_label"]] = {
            "pass": row_case["pass"],
            "wallclock_s": row_case["wallclock_s"],
            "completion_tok_s": row_case["completion_tok_s"],
        }
    ranked = sorted(case_map.items(), key=lambda item: item[1]["wallclock_s"])
    faster_backend = ranked[0][0] if ranked else None
    comparison.append({"case_id": case_id, "backends": case_map, "faster_backend": faster_backend})

output = {
    "report_kind": "psionic_hermes_backend_benchmark",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": psionic_row["host"],
    "psionic_revision": psionic_row["psionic_revision"],
    "hermes_revision": psionic_row["hermes_revision"],
    "fixed_contract": {
        "api_mode": "chat_completions",
        "same_host": True,
        "same_case_ids": case_ids,
        "temperature": 0,
        "seed": 0,
        "tool_policy": "required_then_auto for tool cases; auto for no-tool case",
        "benchmark_boundary": "Hermes fixed; backend swaps only base URL plus backend-specific model identifier",
    },
    "startup_probe": {
        "psionic_ready_s": psionic_ready_end - psionic_ready_start,
        "ollama_ready_s": ollama_ready_end - ollama_ready_start,
    },
    "backends": rows,
    "comparison": comparison,
}
if enable_llamacpp:
    output["startup_probe"]["llama_cpp_ready_s"] = llamacpp_ready_end - llamacpp_ready_start
report_path.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
PY
