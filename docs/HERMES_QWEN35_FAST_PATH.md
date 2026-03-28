# Hermes Qwen3.5 Fast Path

> Status: `implemented_early` on 2026-03-28 for retained qwen35
> `chat.completions` fast-path-versus-fallback truth on exact pushed Psionic.

This document records the retained qwen35 tool-turn fast-path proof for native
Psionic on the OpenAI-compatible `/v1/chat/completions` surface.

The point of this proof is narrower than the broader Hermes compatibility and
backend benchmark docs:

- it does not ask whether Hermes can use Psionic at all
- it does not ask whether Psionic beats Ollama
- it asks whether the current qwen35 CUDA lane stays on the bounded decode path
  for the supported Hermes-style tool turns, and where it still leaves that
  lane

## Exact Revision

- Psionic revision:
  `c417d6aeca6760d143a2efbd9f1c72642bab9c01`
- Host:
  `archlinux`
- Model:
  `qwen3.5-2b-q8_0-registry.gguf`

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/release/check-psionic-hermes-qwen35-fast-path.sh
```

Exact retained command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-fast-path \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_FAST_PATH_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-fastpath-c417d6ae/fixtures/qwen35/hermes/hermes_qwen35_fast_path_benchmark_20260328_archlinux_2b.json \
scripts/release/check-psionic-hermes-qwen35-fast-path.sh
```

## Retained Report

- `fixtures/qwen35/hermes/hermes_qwen35_fast_path_benchmark_20260328_archlinux_2b.json`

## Supported Envelope

The retained report proves a green bounded fast path for:

- `chat.completions`
- debug-field-inclusive replies with `PSIONIC_OPENAI_INCLUDE_DEBUG_FIELDS=1`
- `temperature = 0`, `seed = 0`
- `tool_choice = required` or `tool_choice = auto`
- assistant tool-call replay plus `role = tool` result replay
- bounded qwen35 CUDA decode modes
  - `argmax_only`
  - `top_k_candidates:*`
  - `sparse_logits:*`

The report also makes the current boundaries explicit:

- same-turn parallel tool use is still outside the supported envelope on this
  local qwen35 row because the model emits only one tool call
- `mirostat` currently leaves the bounded lane and materializes dense
  `raw_logits`

## Retained Result

`fast_path_health` on the retained row is:

- `supported_case_count = 3`
- `supported_cases_all_bounded = true`
- `dense_fallback_case_count = 1`
- `outside_envelope_case_count = 2`
- `current_status = bounded_fast_path_green`

Per-case truth:

| Case | Result | Path | Honest reading |
| --- | --- | --- | --- |
| `required_tool_turn_fast_path` | `pass` | `bounded_fast_path` | one required tool call stays on bounded CUDA decode |
| `auto_message_fast_path` | `pass` | `bounded_fast_path` | direct answer stays on bounded CUDA decode |
| `tool_result_continuation_fast_path` | `pass` | `bounded_fast_path` | replayed assistant tool call plus `role = tool` result stays bounded |
| `parallel_tool_turn_model_boundary` | `pass` | `bounded_fast_path` | request surface works, but model still emits only one same-turn tool |
| `mirostat_tool_turn_dense_fallback` | `pass` | `dense_fallback` | qwen35 leaves bounded decode and materializes dense logits |

The important contrast is the readback cost:

- bounded tool rows read back only `9728` to `12288` bytes
- the retained `mirostat` boundary row reads back `156938240` bytes and sets
  `raw_logits_materialized = true`

## What Changed

The retained proof now uses a dedicated runner and direct OpenAI-compatible
probe:

- `scripts/release/check-psionic-hermes-qwen35-fast-path.sh`
- `scripts/release/hermes_qwen35_fast_path_probe.py`

The probe is intentionally not a Hermes-library wrapper. It talks directly to
`/v1/chat/completions` so the retained JSON can capture:

- response headers such as `x-psionic-backend` and `x-psionic-performance-class`
- `psionic_metrics.qwen35_cuda_decode`
- exact output-mode selection
- honest classification into `bounded_fast_path` versus `dense_fallback`

That gives the repo one retained receipt for the operational question that had
been implicit before: whether a Hermes-style tool turn is still on the real
bounded CUDA lane.

## Honest Bottom Line

On exact pushed `c417d6ae`, native Psionic qwen35 now has one retained fast-path
proof showing:

- required tool turns are bounded
- direct auto-answer turns are bounded
- tool-result continuation is bounded
- same-turn parallel tools are still a model-behavior boundary on this row
- `mirostat` is still a real dense-fallback path

That is enough to close the fast-path issue honestly. It is not enough to claim
full Hermes readiness or backend leadership by itself.
