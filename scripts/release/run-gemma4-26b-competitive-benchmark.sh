#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/release/run-gemma4-26b-competitive-benchmark.sh [options]

Options:
  --model-path PATH        GGUF artifact to benchmark
  --psionic-backend NAME   Psionic single-node backend (default: metal on macOS, cuda elsewhere)
  --llama-server PATH      llama.cpp llama-server binary
  --llama-cli PATH         Deprecated alias for --llama-server; if pointed at llama-cli, the sibling
                           llama-server binary is used instead
  --ollama-base-url URL    Ollama base URL (default: http://127.0.0.1:11434)
  --max-tokens N           Output cap shared across all engines (default: 64)
  --ctx-size N             llama.cpp context size (default: 4096)
  --gpu-layers N           llama.cpp gpu layer count (default: 999)
  --out-dir DIR            Benchmark output root (default: fixtures/gemma4/benchmarks)
  --prompt-id ID           Stable prompt identifier (default: gemma4_26b_single_node_tradeoffs_v1)
  --allow-fail             Exit 0 even when the competitive gate verdict is fail
  -h, --help               Show this help and exit

Notes:
  - The runner uses one shared raw-text prompt and one shared GGUF artifact across Psionic,
    Ollama, and llama.cpp.
  - The llama.cpp comparator runs through a temporary local `llama-server` and consumes the
    machine-readable `/completion` timings response instead of scraping CLI stderr.
  - Psionic runs through `gemma4_bench` and must fail closed if the receipt claims native sparse
    execution while any sparse layer actually fell back to host execution.
  - The script writes per-engine receipts plus one merged comparison receipt under the selected
    output directory.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

platform="$(uname -s)"
case "$platform" in
  Darwin)
    default_model_path="/Users/christopherdavid/models/gemma4/all/gemma-4-26B-A4B-it-Q4_K_M.gguf"
    default_psionic_backend="metal"
    ;;
  *)
    default_model_path="/home/christopherdavid/models/gemma4/all/gemma-4-26B-A4B-it-Q4_K_M.gguf"
    default_psionic_backend="cuda"
    ;;
esac

pick_llama_server_default() {
  local candidates=(
    "${PSIONIC_LLAMA_CPP_SERVER:-}"
    "/Users/christopherdavid/work/competition/repos/llama.cpp/build-metal/bin/llama-server"
    "/Users/christopherdavid/work/competition/repos/llama.cpp/build/bin/llama-server"
    "/home/christopherdavid/work/competition/repos/llama.cpp/build/bin/llama-server"
    "/Users/christopherdavid/code/llama.cpp/build/bin/llama-server"
    "/home/christopherdavid/code/llama.cpp/build/bin/llama-server"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf '%s\n' "/Users/christopherdavid/work/competition/repos/llama.cpp/build-metal/bin/llama-server"
}

normalize_llama_server_path() {
  local candidate="$1"
  if [[ -z "$candidate" ]]; then
    return
  fi
  if [[ "$(basename "$candidate")" == "llama-cli" ]]; then
    local sibling
    sibling="$(cd "$(dirname "$candidate")" && pwd)/llama-server"
    if [[ -x "$sibling" ]]; then
      printf '%s\n' "$sibling"
      return
    fi
  fi
  printf '%s\n' "$candidate"
}

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "missing required command: $name" >&2
    exit 1
  fi
}

sanitize_id() {
  local value="$1"
  value="${value//:/_}"
  value="${value//\//_}"
  value="${value// /_}"
  printf '%s' "$value"
}

pick_free_port() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

unix_time_s() {
  python3 - <<'PY'
import time

print(f"{time.time():.6f}")
PY
}

seconds_between() {
  python3 - "$1" "$2" <<'PY'
import sys

start = float(sys.argv[1])
end = float(sys.argv[2])
print(f"{max(0.0, end - start):.6f}")
PY
}

wait_for_http_ready() {
  local url="$1"
  local pid="$2"
  local deadline_seconds="${3:-180}"
  local elapsed=0
  while (( elapsed < deadline_seconds )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "llama.cpp server exited before becoming ready" >&2
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "timed out waiting for llama.cpp server health endpoint: $url" >&2
  return 1
}

model_path="$default_model_path"
psionic_backend="$default_psionic_backend"
llama_server="$(normalize_llama_server_path "$(pick_llama_server_default)")"
ollama_base_url="http://127.0.0.1:11434"
max_tokens=64
ctx_size=4096
gpu_layers=999
out_dir="$repo_root/fixtures/gemma4/benchmarks"
prompt_id="gemma4_26b_single_node_tradeoffs_v1"
allow_fail=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      model_path="${2:?missing value for --model-path}"
      shift 2
      ;;
    --psionic-backend)
      psionic_backend="${2:?missing value for --psionic-backend}"
      shift 2
      ;;
    --llama-server)
      llama_server="$(normalize_llama_server_path "${2:?missing value for --llama-server}")"
      shift 2
      ;;
    --llama-cli)
      llama_server="$(normalize_llama_server_path "${2:?missing value for --llama-cli}")"
      shift 2
      ;;
    --ollama-base-url)
      ollama_base_url="${2:?missing value for --ollama-base-url}"
      shift 2
      ;;
    --max-tokens)
      max_tokens="${2:?missing value for --max-tokens}"
      shift 2
      ;;
    --ctx-size)
      ctx_size="${2:?missing value for --ctx-size}"
      shift 2
      ;;
    --gpu-layers)
      gpu_layers="${2:?missing value for --gpu-layers}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --prompt-id)
      prompt_id="${2:?missing value for --prompt-id}"
      shift 2
      ;;
    --allow-fail)
      allow_fail=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unrecognized argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command cargo
require_command curl
require_command jq
require_command ollama
require_command python3

if [[ ! -f "$model_path" ]]; then
  echo "missing model artifact: $model_path" >&2
  exit 1
fi
if [[ ! -x "$llama_server" ]]; then
  echo "missing llama.cpp server binary: $llama_server" >&2
  exit 1
fi

prompt="$(cat <<'EOF'
You are reviewing an inference engine change for single-node sparse Gemma 4 execution. Explain the tradeoffs between single-node sparse execution and distributed sparse execution for a 26B A4B Gemma model. Focus on latency, memory pressure, expert routing overhead, and operational simplicity.

Then give a short recommendation for when a local single-node Metal path is good enough for a developer workflow, and when a distributed path is still necessary.

Keep the answer concrete and technical. Use plain language and avoid marketing phrasing.
EOF
)"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
host_label="$(hostname -s | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')"
run_id="gemma4_26b_competitive_${timestamp}_$(sanitize_id "$host_label")"
report_dir="$out_dir/reports/$run_id"
raw_dir="$report_dir/raw"
comparison_path="$out_dir/${run_id}.json"
mkdir -p "$raw_dir"

tmp_dir="$(mktemp -d)"
ollama_model="psionic-gemma4-26b-bench-${timestamp}-$$"
ollama_modelfile="$tmp_dir/Modelfile"
printf 'FROM %s\n' "$model_path" >"$ollama_modelfile"
llama_port="$(pick_free_port)"
llama_base_url="http://127.0.0.1:${llama_port}"
llama_server_pid=""

cleanup() {
  if [[ -n "${llama_server_pid:-}" ]]; then
    kill "$llama_server_pid" >/dev/null 2>&1 || true
    wait "$llama_server_pid" >/dev/null 2>&1 || true
  fi
  ollama stop "$ollama_model" >/dev/null 2>&1 || true
  ollama rm "$ollama_model" >/dev/null 2>&1 || true
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

psionic_report="$report_dir/psionic.json"
ollama_raw="$raw_dir/ollama_generate.json"
ollama_create_stderr="$raw_dir/ollama_create_stderr.txt"
ollama_report="$report_dir/ollama.json"
llama_raw="$raw_dir/llama_completion.json"
llama_server_stdout="$raw_dir/llama_server_stdout.txt"
llama_server_stderr="$raw_dir/llama_server_stderr.txt"
llama_report="$report_dir/llama_cpp.json"

ollama_base_url="${ollama_base_url%/}"

echo "running Psionic gemma4 benchmark..."
cargo run --release -q -p psionic-serve --example gemma4_bench -- \
  --model-path "$model_path" \
  --mode single \
  --backend "$psionic_backend" \
  --prompt-id "$prompt_id" \
  --prompt-mode text \
  --prompt "$prompt" \
  --max-output-tokens "$max_tokens" \
  --repeats 1 \
  --json-out "$psionic_report"

echo "creating temporary Ollama model ${ollama_model}..."
ollama create "$ollama_model" -f "$ollama_modelfile" >/dev/null 2>"$ollama_create_stderr"

echo "running Ollama comparator..."
curl -fsS "${ollama_base_url}/api/generate" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc \
    --arg model "$ollama_model" \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model,prompt:$prompt,raw:true,stream:false,options:{temperature:0,num_predict:$max_tokens}}'
  )" \
  >"$ollama_raw"

echo "starting llama.cpp comparator server on ${llama_base_url}..."
llama_server_started_at="$(unix_time_s)"
"$llama_server" \
  --model "$model_path" \
  --host 127.0.0.1 \
  --port "$llama_port" \
  --ctx-size "$ctx_size" \
  --gpu-layers "$gpu_layers" \
  --parallel 1 \
  --threads-http 4 \
  --perf \
  >"$llama_server_stdout" 2>"$llama_server_stderr" &
llama_server_pid="$!"
wait_for_http_ready "${llama_base_url}/health" "$llama_server_pid"
llama_server_ready_at="$(unix_time_s)"
llama_startup_s="$(seconds_between "$llama_server_started_at" "$llama_server_ready_at")"

echo "running llama.cpp comparator..."
curl -fsS "${llama_base_url}/completion" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{prompt:$prompt,stream:false,n_predict:$max_tokens,temperature:0,top_k:1,top_p:1,cache_prompt:false}'
  )" \
  >"$llama_raw"
kill "$llama_server_pid" >/dev/null 2>&1 || true
wait "$llama_server_pid" >/dev/null 2>&1 || true
llama_server_pid=""

echo "assembling comparison receipt..."
python3 - "$psionic_report" "$ollama_raw" "$ollama_report" "$llama_raw" "$llama_report" "$comparison_path" "$prompt_id" "$model_path" "$max_tokens" "$llama_server" "$llama_startup_s" "$ollama_model" "$report_dir" "$allow_fail" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

psionic_report_path = Path(sys.argv[1])
ollama_raw_path = Path(sys.argv[2])
ollama_report_path = Path(sys.argv[3])
llama_raw_path = Path(sys.argv[4])
llama_report_path = Path(sys.argv[5])
comparison_path = Path(sys.argv[6])
prompt_id = sys.argv[7]
model_path = Path(sys.argv[8])
max_tokens = int(sys.argv[9])
llama_server = sys.argv[10]
llama_startup_s = float(sys.argv[11])
ollama_model = sys.argv[12]
report_dir = Path(sys.argv[13])
allow_fail = sys.argv[14] == "1"


def ns_to_s(value):
    if value is None:
        return None
    return float(value) / 1_000_000_000.0


def tok_s(tokens, seconds):
    if not tokens or not seconds:
        return None
    if seconds <= 0:
        return None
    return float(tokens) / float(seconds)


def read_json(path: Path):
    return json.loads(path.read_text())


def maybe_cmd_output(cmd):
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return output.strip()
    except Exception:
        return None


def repo_relative(path: Path):
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def normalize_llama_cpp_version(output):
    if output is None:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    version_lines = [
        line for line in lines if line.startswith("version:") or line.startswith("built with ")
    ]
    if version_lines:
        return "\n".join(version_lines)
    return output


def parse_llama_completion(raw: dict):
    timings = raw.get("timings") or {}
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    predicted_n = timings.get("predicted_n")
    if prompt_ms is None or predicted_ms is None or predicted_n is None:
        raise SystemExit(
            f"failed to parse llama.cpp timings from {llama_raw_path}: missing prompt_ms, predicted_ms, or predicted_n"
        )
    return {
        "load_s": llama_startup_s,
        "prompt_eval": float(prompt_ms) / 1000.0,
        "decode": float(predicted_ms) / 1000.0,
        "eval_count": int(predicted_n),
        "total": llama_startup_s + (float(prompt_ms) + float(predicted_ms)) / 1000.0,
        "predicted_per_second": timings.get("predicted_per_second"),
    }


psionic = read_json(psionic_report_path)
ollama_raw = read_json(ollama_raw_path)
llama_raw = read_json(llama_raw_path)
llama_timings = parse_llama_completion(llama_raw)

ollama = {
    "schema_version": 1,
    "report_kind": "ollama_gemma4_26b_bench",
    "model_artifact": model_path.name,
    "model_name": ollama_model,
    "benchmark_prompt_id": prompt_id,
    "max_output_tokens": max_tokens,
    "load_s": ns_to_s(ollama_raw.get("load_duration")),
    "prompt_eval": ns_to_s(ollama_raw.get("prompt_eval_duration")),
    "decode": ns_to_s(ollama_raw.get("eval_duration")),
    "total": ns_to_s(ollama_raw.get("total_duration")),
    "decode_tok_s": tok_s(
        ollama_raw.get("eval_count"),
        ns_to_s(ollama_raw.get("eval_duration")),
    ),
    "output_tokens": ollama_raw.get("eval_count"),
    "termination": ollama_raw.get("done_reason"),
    "output_text": ollama_raw.get("response", ""),
    "state_readback_bytes": None,
}

llama_cpp = {
    "schema_version": 1,
    "report_kind": "llama_cpp_gemma4_26b_bench",
    "model_artifact": model_path.name,
    "llama_server": str(llama_server),
    "benchmark_prompt_id": prompt_id,
    "max_output_tokens": max_tokens,
    "load_s": llama_timings["load_s"],
    "prompt_eval": llama_timings["prompt_eval"],
    "decode": llama_timings["decode"],
    "total": llama_timings["total"],
    "decode_tok_s": llama_timings["predicted_per_second"]
    if llama_timings["predicted_per_second"] is not None
    else tok_s(llama_timings["eval_count"], llama_timings["decode"]),
    "output_tokens": llama_timings["eval_count"],
    "termination": llama_raw.get("stop_type"),
    "output_text": llama_raw.get("content", ""),
    "state_readback_bytes": None,
}

ollama_report_path.write_text(json.dumps(ollama, indent=2) + "\n")
llama_report_path.write_text(json.dumps(llama_cpp, indent=2) + "\n")

psionic_decode_tok_s = psionic.get("decode_tok_s")
ollama_decode_tok_s = ollama.get("decode_tok_s")
llama_decode_tok_s = llama_cpp.get("decode_tok_s")

reasons = []
host_fallback_observed = psionic.get("host_fallback_observed")
native_sparse_execution = psionic.get("native_sparse_execution")
sparse_ffn_backend = psionic.get("sparse_ffn_backend")

if native_sparse_execution is True and host_fallback_observed:
    reasons.append("Psionic receipt claims native sparse execution while host fallback was observed")
if native_sparse_execution is True and isinstance(sparse_ffn_backend, str) and "host_fallback" in sparse_ffn_backend:
    reasons.append("Psionic receipt claims native sparse execution while sparse_ffn_backend includes host_fallback")
if host_fallback_observed is None:
    reasons.append("Psionic receipt is missing host_fallback_observed")
if sparse_ffn_backend is None:
    reasons.append("Psionic receipt is missing sparse_ffn_backend")

if psionic_decode_tok_s is None:
    reasons.append("Psionic receipt is missing decode_tok_s")
if ollama_decode_tok_s is None:
    reasons.append("Ollama receipt is missing decode_tok_s")
if llama_decode_tok_s is None:
    reasons.append("llama.cpp receipt is missing decode_tok_s")

beats_ollama = (
    psionic_decode_tok_s is not None
    and ollama_decode_tok_s is not None
    and psionic_decode_tok_s >= ollama_decode_tok_s
)
beats_llama_cpp = (
    psionic_decode_tok_s is not None
    and llama_decode_tok_s is not None
    and psionic_decode_tok_s >= llama_decode_tok_s
)
if not beats_ollama:
    reasons.append("Psionic decode_tok_s is below the same-host Ollama baseline")
if not beats_llama_cpp:
    reasons.append("Psionic decode_tok_s is below the same-host llama.cpp baseline")

comparison = {
    "schema_version": 1,
    "report_kind": "gemma4_26b_competitive_benchmark",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "host": platform.node(),
    "git_commit": maybe_cmd_output(["git", "rev-parse", "HEAD"]),
    "model_path": str(model_path),
    "model_artifact": model_path.name,
    "benchmark_prompt_id": prompt_id,
    "max_output_tokens": max_tokens,
    "ollama_version": maybe_cmd_output(["ollama", "--version"]),
    "llama_cpp_version": normalize_llama_cpp_version(
        maybe_cmd_output([str(llama_server), "--version"])
    ),
    "reports": {
        "psionic": repo_relative(psionic_report_path),
        "ollama": repo_relative(ollama_report_path),
        "llama_cpp": repo_relative(llama_report_path),
        "report_dir": repo_relative(report_dir),
    },
    "psionic": psionic,
    "ollama": ollama,
    "llama_cpp": llama_cpp,
    "gate": {
        "pass": len(reasons) == 0,
        "allow_fail": allow_fail,
        "beats_ollama": beats_ollama,
        "beats_llama_cpp": beats_llama_cpp,
        "fail_closed_sparse_receipt": not (
            native_sparse_execution is True and host_fallback_observed
        ),
        "reasons": reasons,
    },
}

comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison["gate"], indent=2))
if reasons and not allow_fail:
    raise SystemExit(1)
PY

echo "wrote comparison receipt: $comparison_path"
