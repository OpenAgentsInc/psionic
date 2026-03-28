# Hermes Qwen3.5 Compatibility

> Status: `implemented_early` on 2026-03-28 for retained Hermes-on-Psionic
> `chat.completions` compatibility evidence.

This document records the retained Hermes compatibility proof for native
Psionic `qwen35` on the OpenAI-compatible `/v1/chat/completions` surface.

The proof is honest about the current boundary:

- Hermes-on-Psionic is real for `5/6` retained cases on the exact pushed
  revision below.
- The only remaining failed case is same-turn parallel tool calling.
- The retained request summaries prove Hermes sent both declared tools, so the
  remaining blocker is current qwen35 model behavior on these local rows, not a
  missing tool-surface transport path in Psionic.

## Exact Revisions

- Psionic revision:
  `ef5e2cdca840db6b2fc0c871649e6cb4b2af6d30`
- Hermes revision:
  `e295a2215acd55f2ee930fc7a4cd2df1c5464234`
- Host:
  `archlinux`

## Canonical Checker

Run the retained checker from the repo root:

```bash
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

The retained proof used the exact pushed Psionic revision above and the Hermes
checkout above, with explicit overrides for the remote binary, model path,
report path, and scratch `TMPDIR`.

Exact 2B proof command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-check \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-proof-ef5e2cdc/fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b.json \
PSIONIC_HERMES_SERVER_LOG_PATH=/home/christopherdavid/scratch/logs/hermes_qwen35_psionic_compatibility_20260328_archlinux_2b.log \
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

Exact 9B proof command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-check \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf \
PSIONIC_HERMES_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-proof-ef5e2cdc/fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_9b.json \
PSIONIC_HERMES_SERVER_LOG_PATH=/home/christopherdavid/scratch/logs/hermes_qwen35_psionic_compatibility_20260328_archlinux_9b.log \
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

## Retained Reports

- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_9b.json`

## Retained Matrix

| Model | Pass Count | Overall | Failed Case |
| --- | --- | --- | --- |
| `qwen3.5-2b-q8_0-registry.gguf` | `5/6` | `false` | `parallel_tool_turn` |
| `qwen3.5-9b-q4_k_m-registry.gguf` | `5/6` | `false` | `parallel_tool_turn` |

Both retained rows pass:

- `required_tool_turn`
- `auto_plain_text_turn`
- `multi_turn_tool_loop`
- `invalid_argument_truthful_refusal`
- `streamed_tool_turn`

Both retained rows fail:

- `parallel_tool_turn`

The first same-host backend benchmark now lives in
`docs/HERMES_BACKEND_BENCHMARK.md`.

## What The Proof Actually Shows

The retained reports prove the following source-level compatibility work is now
real on the native Psionic qwen35 lane:

- leading `system` plus `developer` messages are normalized into one qwen35
  instruction prelude instead of being dropped or reordered
- assistant tool-call turns can be replayed back into qwen35 as the raw tool
  envelope the model expects
- `role = tool` result turns can be replayed without losing `tool_call_id`
  linkage, including name recovery from prior assistant tool calls
- Hermes can use the normal OpenAI-compatible `chat.completions` path against
  Psionic rather than a custom side channel
- streamed tool-call deltas and the request-level `parallel_tool_calls` field
  are admitted on the Psionic surface

## Remaining Blocker

The current remaining blocker is not missing request-surface plumbing.

In both retained failing rows, the request summaries show Hermes sent both tool
definitions on the parallel turn:

- `get_paris_weather`
- `get_tokyo_weather`

But the assistant emitted only one tool call:

- `get_paris_weather`

That means the current red case is the local qwen35 model response behavior on
same-turn parallel tool use, not that Psionic failed to expose multiple tools,
`parallel_tool_calls`, tool replay, or tool result ingestion.

## Honest Bottom Line

Psionic is now materially closer to real Hermes readiness than the earlier
audit posture implied:

- source and retained receipts now prove `5/6` Hermes compatibility on exact
  pushed Psionic for local qwen35 rows
- the remaining blocker is sharply bounded to same-turn parallel tool calling
- the repo now also has one retained Psionic-versus-Ollama Hermes benchmark on
  the local `2b` row
- this is still not enough to claim full Hermes compatibility yet
