# Hermes Backend Benchmark

> Status: `implemented_early` on 2026-03-28 for same-host Hermes benchmark
> evidence against Psionic, Ollama, and an optional `llama.cpp` comparator
> lane.

This document records the retained same-host Hermes backend benchmark for local
`qwen35`, keeping Hermes fixed and swapping only the OpenAI-compatible backend
endpoint and backend-specific model identifier.

The repo now retains two benchmark receipts:

- the first Psionic-vs-Ollama benchmark at
  `473c0ad5bd5219cbcb7f76495d8166933242b872`
- the optional three-way follow-on at
  `477ca8226589a5b1760ed93c47b87041f08172ab`

The later receipt is the canonical answer for the optional `llama.cpp`
comparator issue because it preserves the same benchmark contract and records
the third lane honestly even when that lane is unavailable on the current host.

## Exact Revisions

- Psionic revision:
  `477ca8226589a5b1760ed93c47b87041f08172ab`
- Hermes revision:
  `e295a2215acd55f2ee930fc7a4cd2df1c5464234`
- Host:
  `archlinux`
- `llama-server` version:
  `7444 (58062860a)`

## Canonical Runner

Run the benchmark from the repo root:

```bash
scripts/release/run-hermes-backend-benchmark.sh
```

The exact retained three-way receipt used:

- Psionic model path:
  `/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf`
- Ollama model:
  `qwen3.5:2b`
- Hermes root:
  `/home/christopherdavid/scratch/hermes-agent-proof2`
- Hermes Python:
  `/home/christopherdavid/scratch/hermes-min/.venv/bin/python`
- Psionic server binary:
  `/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server`
- `llama.cpp` server binary:
  `/home/christopherdavid/code/llama.cpp/build/bin/llama-server`

Exact retained command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-bench \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_PSIONIC_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_OLLAMA_MODEL=qwen3.5:2b \
PSIONIC_HERMES_ENABLE_LLAMACPP=1 \
PSIONIC_HERMES_LLAMA_CPP_MODEL_ALIAS=qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_BENCHMARK_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-benchmark-477ca822/fixtures/qwen35/hermes/hermes_psionic_vs_ollama_vs_llamacpp_benchmark_20260328_archlinux_2b.json \
PSIONIC_HERMES_BENCHMARK_RAW_DIR=/home/christopherdavid/scratch/psionic-hermes-benchmark-477ca822/fixtures/qwen35/hermes/backend_rows \
scripts/release/run-hermes-backend-benchmark.sh
```

## Fixed Contract

This benchmark keeps the following fixed:

- same host
- same Hermes revision
- same Hermes `chat.completions` custom-provider path
- same case ids
- same tool schemas and handlers
- `temperature = 0`
- `seed = 0`
- same `required_then_auto` tool policy for tool cases
- same `auto` policy for the no-tool case

What changed between rows:

- `OPENAI_BASE_URL`
- backend-specific model identifier
  - Psionic: local GGUF basename
  - Ollama: local Ollama model name
  - `llama.cpp`: local GGUF basename

## Retained Reports

- first two-way aggregate:
  `fixtures/qwen35/hermes/hermes_psionic_vs_ollama_benchmark_20260328_archlinux_2b.json`
- three-way aggregate:
  `fixtures/qwen35/hermes/hermes_psionic_vs_ollama_vs_llamacpp_benchmark_20260328_archlinux_2b.json`
- three-way Psionic row:
  `fixtures/qwen35/hermes/backend_rows/hermes_psionic_row_20260328_archlinux_2b_llamacpp_comparator.json`
- three-way Ollama row:
  `fixtures/qwen35/hermes/backend_rows/hermes_ollama_row_20260328_archlinux_2b_llamacpp_comparator.json`
- three-way `llama.cpp` row:
  `fixtures/qwen35/hermes/backend_rows/hermes_llama_cpp_row_20260328_archlinux_2b.json`

## Cases

The retained benchmark uses four Hermes cases:

- `auto_plain_text_turn`
- `required_tool_turn`
- `multi_turn_tool_loop`
- `streamed_tool_turn`

Psionic and Ollama both passed all four retained cases on the exact
`477ca822` rerun.

## Current Same-Host Result

| Case | Psionic wallclock s | Ollama wallclock s | llama.cpp | Faster |
| --- | ---: | ---: | --- | --- |
| `required_tool_turn` | `3.1804` | `2.9847` | `startup_failure` | `ollama` |
| `auto_plain_text_turn` | `1.1518` | `0.4791` | `startup_failure` | `ollama` |
| `multi_turn_tool_loop` | `3.1403` | `1.7067` | `startup_failure` | `ollama` |
| `streamed_tool_turn` | `2.9973` | `1.7697` | `startup_failure` | `ollama` |

Per-row summary:

- Psionic:
  - `overall_pass = true`
  - `passing_case_count = 4/4`
  - `mean_case_wallclock_s = 2.6174`
  - `mean_completion_tok_s = null`
- Ollama:
  - `overall_pass = true`
  - `passing_case_count = 4/4`
  - `mean_case_wallclock_s = 1.7351`
  - `mean_completion_tok_s = 88.5821`
- `llama.cpp`:
  - `overall_pass = false`
  - `row_status = startup_failure`
  - `failure_detail = llama.cpp server failed readiness check on 127.0.0.1:8098`

Availability probe on this same host:

- Psionic readiness probe:
  `0.8806s`
- Ollama readiness probe:
  `0.0127s`
- `llama.cpp` readiness probe:
  `null`
- `llama.cpp` startup status:
  `startup_failure`

That readiness probe is not a cold-start apples-to-apples claim. Psionic is
being launched by the harness, while Ollama is measured as an already-running
same-host service.

## Why The llama.cpp Lane Matters

This lane matters only when `llama.cpp` is a real runnable backend for the same
model contract. That is useful for:

- Mac-local CPU or Metal operability questions
- checking whether the same model artifact can cross the Psionic and
  `llama.cpp` boundary without format or tokenizer drift
- separating "Psionic is slower" from "the comparator is not actually runnable
  here"

On this exact `archlinux` `RTX 4080` host, the lane is currently a compatibility
boundary check, not a throughput comparator:

- the local `llama-server` binary can start, but it cannot load the retained
  rewritten Psionic `qwen35` GGUF
- the retained `llama.cpp` row captures the exact failure:
  `unknown model architecture: 'qwen35'`
- `ollama show --modelfile qwen3.5:2b` points at
  `/usr/share/ollama/.ollama/models/blobs/sha256-b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297`
- that blob path is permission-blocked to the current user, so the harness
  cannot honestly reroute the Ollama-managed artifact through `llama.cpp` on
  this host

That is why the wrapper now retains a synthetic `llama_cpp` row instead of
crashing the whole benchmark. The comparator boundary is now explicit,
machine-readable, and rerunnable.

## Honest Interpretation

The retained three-way receipt does **not** show usable `llama.cpp` throughput
for this `qwen3.5` lane. It shows:

- Psionic and Ollama both complete the same four Hermes cases on the same host
  under the same high-level contract
- Ollama still wins wallclock on all four retained cases
- the optional `llama.cpp` comparator is wired into the same harness, but the
  current host and model contract stop it at startup

That is enough to close the optional comparator issue honestly. The repo no
longer has a silent third-lane gap or a benchmark wrapper that aborts as soon
as the comparator cannot start.

## Current Gaps

Two real gaps remain visible after the three-way receipt:

- Psionic is still slower than Ollama on all four current same-host Hermes
  cases on this `qwen3.5` `2b` row
- Psionic still does not expose comparable usage accounting on this local
  Hermes benchmark lane, so `completion_tok_s` remains `null`

The optional `llama.cpp` lane also remains blocked for real throughput
comparison until one of these becomes true:

- `llama.cpp` gains support for the rewritten `qwen35` GGUF contract
- the benchmark can access a `llama.cpp`-loadable model artifact on the same
  host without permission tricks or hidden format swaps

## Relation To Other Hermes Issues

- direct compatibility proof remains in `docs/HERMES_QWEN35_COMPATIBILITY.md`
- fast-path-versus-fallback runtime truth now lives in
  `docs/HERMES_QWEN35_FAST_PATH.md`
- session reuse and prefix-cache reuse belong to the later repeated-session
  latency issue
