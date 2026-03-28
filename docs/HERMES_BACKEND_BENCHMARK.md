# Hermes Backend Benchmark

> Status: `implemented_early` on 2026-03-28 for same-host Hermes benchmark
> evidence against Psionic and Ollama.

This document records the first retained same-host Hermes backend benchmark for
local `qwen35`, keeping Hermes fixed and swapping only the OpenAI-compatible
backend endpoint and backend-specific model identifier.

## Exact Revisions

- Psionic revision:
  `473c0ad5bd5219cbcb7f76495d8166933242b872`
- Hermes revision:
  `e295a2215acd55f2ee930fc7a4cd2df1c5464234`
- Host:
  `archlinux`

## Canonical Runner

Run the benchmark from the repo root:

```bash
scripts/release/run-hermes-backend-benchmark.sh
```

The retained benchmark used:

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

Exact retained command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-bench \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_PSIONIC_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_OLLAMA_MODEL=qwen3.5:2b \
PSIONIC_HERMES_BENCHMARK_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-benchmark-473c0ad5/fixtures/qwen35/hermes/hermes_psionic_vs_ollama_benchmark_20260328_archlinux_2b.json \
PSIONIC_HERMES_BENCHMARK_RAW_DIR=/home/christopherdavid/scratch/psionic-hermes-benchmark-473c0ad5/fixtures/qwen35/hermes/backend_rows \
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

What changed between the two rows:

- `OPENAI_BASE_URL`
- backend-specific model identifier
  - Psionic: local GGUF basename
  - Ollama: local Ollama model name

## Retained Reports

- aggregate report:
  `fixtures/qwen35/hermes/hermes_psionic_vs_ollama_benchmark_20260328_archlinux_2b.json`
- Psionic row:
  `fixtures/qwen35/hermes/backend_rows/hermes_psionic_row_20260328_archlinux_2b.json`
- Ollama row:
  `fixtures/qwen35/hermes/backend_rows/hermes_ollama_row_20260328_archlinux_2b.json`

## Cases

The retained benchmark uses four Hermes cases:

- `auto_plain_text_turn`
- `required_tool_turn`
- `multi_turn_tool_loop`
- `streamed_tool_turn`

Both backends passed all four retained cases on this rerun.

## Current Same-Host Result

| Case | Psionic wallclock s | Ollama wallclock s | Faster |
| --- | ---: | ---: | --- |
| `required_tool_turn` | `3.2143` | `2.9274` | `ollama` |
| `auto_plain_text_turn` | `1.1450` | `0.4922` | `ollama` |
| `multi_turn_tool_loop` | `3.0261` | `1.7262` | `ollama` |
| `streamed_tool_turn` | `3.0484` | `1.7710` | `ollama` |

Per-row summary:

- Psionic:
  - `overall_pass = true`
  - `passing_case_count = 4/4`
  - `mean_case_wallclock_s = 2.6085`
  - `mean_completion_tok_s = null`
- Ollama:
  - `overall_pass = true`
  - `passing_case_count = 4/4`
  - `mean_case_wallclock_s = 1.7292`
  - `mean_completion_tok_s = 85.1574`

Availability probe on this same host:

- Psionic readiness probe:
  `0.9651s`
- Ollama readiness probe:
  `0.0127s`

That readiness probe is not a full cold-start apples-to-apples claim. Psionic is
being launched by the harness, while Ollama is measured as an already-running
same-host service.

## Honest Interpretation

This first retained same-host Hermes benchmark does **not** show Psionic ahead
of Ollama yet on the 2B row. Ollama won wallclock on all four retained cases.

The retained receipt does still prove useful things:

- the repo now has one rerunnable Hermes benchmark harness instead of only a
  functional compatibility checker
- Psionic and Ollama both complete the same four Hermes cases on the same host
  under the same high-level contract
- the benchmark is already sensitive enough to catch real runtime problems; an
  earlier pre-fix rerun exposed a streamed-turn continuation failure on the
  Psionic row and forced the harness to tighten its pass truth

## Current Gaps

Two real gaps remain visible in the retained receipt:

- Psionic is slower than Ollama on all four current same-host Hermes cases on
  this `qwen3.5` `2b` benchmark row
- Psionic does not currently expose comparable usage accounting on this local
  Hermes benchmark lane, so its `completion_tok_s` is still `null` in the
  retained row instead of a real measured token rate

That second gap is a measurement-truth problem, not a throughput win. The
benchmark therefore publishes `null` rather than pretending that the missing
usage fields are zero-token generations.

## Relation To Other Hermes Issues

- direct compatibility proof remains in `docs/HERMES_QWEN35_COMPATIBILITY.md`
- the optional third comparator belongs to the later llama.cpp issue
- the fast-path-versus-fallback and runtime-truth work belongs to the later
  qwen35 Hermes performance issue
