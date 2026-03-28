# Hermes Qwen3.5 Compatibility

> Status: `implemented_early` on 2026-03-28 for retained Hermes-on-Psionic
> `chat.completions` compatibility evidence.

This document records the retained Hermes compatibility proof for native
Psionic `qwen35` on the OpenAI-compatible `/v1/chat/completions` surface.

The proof is honest about the current boundary:

- Hermes-on-Psionic is real for `5/6` retained cases on the exact pushed
  revisions below.
- The only remaining failed case is same-turn parallel tool calling.
- The retained request summaries prove Hermes sent both declared tools, so the
  remaining blocker is current qwen35 model behavior on these local rows, not a
  missing tool-surface transport path in Psionic.
- The later local-tailnet follow-up reruns on pushed `b2277ed5` and
  `9f67d65b` stayed at the same `5/6` boundary on the retained `2b` and `9b`
  rows even after tightening both the backend parallel-tool prompt contract and
  the repo-owned compatibility probe.

## Retained Revisions

- Baseline retained proof revision:
  `ef5e2cdca840db6b2fc0c871649e6cb4b2af6d30`
- Backend prompt-tightening follow-up:
  `b2277ed5a16b640285574b6e450d9c772c898c9b`
- Latest local-tailnet checker-tightening follow-up:
  `9f67d65babb940a5e7878c518532b5068b424486`
- Hermes revision:
  `e295a2215acd55f2ee930fc7a4cd2df1c5464234`
- Host:
  `archlinux`

## Canonical Checker

Run the retained checker from the repo root:

```bash
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

The retained proof used the exact pushed Psionic revisions above and the Hermes
checkout above, with explicit overrides for the remote binary, model path,
report path, and scratch `TMPDIR`.

Latest exact 2B follow-up command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-check-9f67d65b \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf \
PSIONIC_HERMES_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-compat-9f67d65b/fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b_9f67d65b.json \
PSIONIC_HERMES_SERVER_LOG_PATH=/home/christopherdavid/scratch/logs/hermes_qwen35_psionic_compatibility_20260328_archlinux_2b_9f67d65b.log \
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

Latest exact 9B follow-up command:

```bash
TMPDIR=/home/christopherdavid/scratch/tmp/hermes-check-9f67d65b-9b \
PSIONIC_HERMES_ROOT=/home/christopherdavid/scratch/hermes-agent-proof2 \
PSIONIC_HERMES_PYTHON=/home/christopherdavid/scratch/hermes-min/.venv/bin/python \
PSIONIC_HERMES_SERVER_BIN=/home/christopherdavid/.cache/psionic-hermes-target/debug/psionic-openai-server \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf \
PSIONIC_HERMES_REPORT_PATH=/home/christopherdavid/scratch/psionic-hermes-compat-9f67d65b/fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_9b_9f67d65b.json \
PSIONIC_HERMES_SERVER_LOG_PATH=/home/christopherdavid/scratch/logs/hermes_qwen35_psionic_compatibility_20260328_archlinux_9b_9f67d65b.log \
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

## Retained Reports

- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b_b2277ed5.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_2b_9f67d65b.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_9b.json`
- `fixtures/qwen35/hermes/hermes_qwen35_psionic_compatibility_report_20260328_archlinux_9b_9f67d65b.json`

## Retained Matrix

| Revision | Model | Pass Count | Overall | Failed Case |
| --- | --- | --- | --- | --- |
| `ef5e2cdc` | `qwen3.5-2b-q8_0-registry.gguf` | `5/6` | `false` | `parallel_tool_turn` |
| `ef5e2cdc` | `qwen3.5-9b-q4_k_m-registry.gguf` | `5/6` | `false` | `parallel_tool_turn` |
| `b2277ed5` | `qwen3.5-2b-q8_0-registry.gguf` | `5/6` | `false` | `parallel_tool_turn` |
| `9f67d65b` | `qwen3.5-2b-q8_0-registry.gguf` | `5/6` | `false` | `parallel_tool_turn` |
| `9f67d65b` | `qwen3.5-9b-q4_k_m-registry.gguf` | `5/6` | `false` | `parallel_tool_turn` |

Every retained row above passes:

- `required_tool_turn`
- `auto_plain_text_turn`
- `multi_turn_tool_loop`
- `invalid_argument_truthful_refusal`
- `streamed_tool_turn`

Every retained row above fails:

- `parallel_tool_turn`

One experimental local `4b` rerun on the older `972661d4` worktree was worse
than the canonical retained rows: it fell to `4/6` because the same parallel
row still failed and the streamed row ended in connection failure. That run is
useful as a local operator signal, but it is not the canonical retained proof
path.

The first same-host backend benchmark now lives in
`docs/HERMES_BACKEND_BENCHMARK.md`.

The later bounded fast-path-versus-fallback proof now lives in
`docs/HERMES_QWEN35_FAST_PATH.md`.

The later repeated-loop prefix-cache proof now lives in
`docs/HERMES_QWEN35_REUSE_BENCHMARK.md`.

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

In every retained failing row, the request summaries show Hermes sent both tool
definitions on the parallel turn:

- `get_paris_weather`
- `get_tokyo_weather`

But the assistant still emitted only one tool call:

- `get_paris_weather`

The latest local follow-up on `9f67d65b` tightened the probe further:

- the parallel-case system message now explicitly requires exactly two tool
  calls in order
- each tool description now states that it is valid only for its own city and
  must not be substituted for the other

Even with those stricter instructions:

- local `2b` still emitted `get_paris_weather`, then repeated
  `get_paris_weather` on the second iteration
- local `9b` still emitted only `get_paris_weather`

That means the current red case is still the local qwen35 model response
behavior on same-turn parallel tool use, not that Psionic failed to expose
multiple tools, `parallel_tool_calls`, tool replay, or tool result ingestion.

## Honest Bottom Line

Psionic is now materially closer to real Hermes readiness than the earlier
audit posture implied:

- source and retained receipts now prove `5/6` Hermes compatibility on exact
  pushed Psionic for local qwen35 rows
- the remaining blocker is sharply bounded to same-turn parallel tool calling
- the repo now also has one retained Psionic-versus-Ollama Hermes benchmark on
  the local `2b` row
- the repo now also has one retained exact-pushed qwen35 fast-path proof for
  required tool turns, direct auto turns, and tool-result continuation
- the repo now also has one retained repeated-loop reuse receipt showing warm
  wallclock improvement on required tool turns and tool-result continuation
- local-tailnet qwen35 improvement work for this exact blocker is now close to
  exhausted: backend prompt tightening, checker tightening, and the available
  `2b`, `4b`, and `9b` local rows all leave the same-turn parallel-tool case
  unresolved
- this is still not enough to claim full Hermes compatibility yet, so the
  direct-compatibility umbrella issue stays honestly open until a stronger row
  or different runtime path clears the last case
