# Hermes On Psionic

This is the user-facing guide for running Hermes against Psionic.

In plain language:

- Psionic can now act as a real backend for Hermes on one consumer GPU.
- Hermes can talk to Psionic through the normal OpenAI-compatible
  `chat.completions` path.
- Hermes can answer normally, call tools, read tool results, keep going across
  multiple turns, stream tool calls, and do a same-turn two-tool batch when
  required.
- Psionic is the backend here. Hermes is still a separate checkout. Psionic
  does not bundle a full Hermes product or CLI.

If you want the shortest honest answer to "can I command Hermes to do shit
through Psionic?":

- yes, if you already have a Hermes checkout
- you point Hermes at a running `psionic-openai-server`
- you use `provider="custom"` and `api_mode="chat_completions"`
- you set the model name to the GGUF basename Psionic is serving

## What Works Today

The current retained consumer-GPU proof is:

- host: `archlinux`
- GPU: `RTX 4080`
- model family: local `qwen3.5`
- canonical direct compatibility result: `6/6`

That means the current native Psionic Hermes lane can do all of the following:

- required tool call
- plain-text no-tool answer
- multi-turn tool loop with tool-result replay
- same-turn two-tool assistant response
- truthful refusal after an invalid tool result
- streamed tool-call turn

Canonical retained proof:

- `docs/HERMES_QWEN35_COMPATIBILITY.md`
- `docs/HERMES_QWEN35_PARALLEL_ATTRIBUTION.md`

## Fastest Proof Path

If you just want to prove your local setup works end to end, run the repo-owned
checker.

From the Psionic repo root:

```bash
PSIONIC_HERMES_ROOT=/abs/path/to/hermes \
PSIONIC_HERMES_PYTHON=/abs/path/to/hermes/.venv/bin/python \
PSIONIC_HERMES_QWEN35_MODEL_PATH=/abs/path/to/qwen3.5-2b-q8_0-registry.gguf \
scripts/release/check-psionic-hermes-qwen35-compatibility.sh
```

That script will:

1. build or reuse `psionic-openai-server`
2. start a local Psionic OpenAI-compatible server
3. point Hermes at it through `OPENAI_BASE_URL`
4. run the retained compatibility matrix
5. write a JSON receipt under `fixtures/qwen35/hermes/`

If that passes, your local Hermes-on-Psionic lane is working.

## How You Actually Use It

### 1. Run Psionic as the backend

Build the server:

```bash
cargo build -p psionic-serve --bin psionic-openai-server --release
```

Run it on a Linux NVIDIA host:

```bash
./target/release/psionic-openai-server \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8095 \
  -m /abs/path/to/qwen3.5-2b-q8_0-registry.gguf
```

You should then have:

- health check: `http://127.0.0.1:8095/health`
- OpenAI-compatible base URL: `http://127.0.0.1:8095/v1`

### 2. Point Hermes at that server

The repo-owned compatibility probe uses Hermes like this:

```python
from run_agent import AIAgent

agent = AIAgent(
    base_url="http://127.0.0.1:8095/v1",
    api_key="dummy",
    provider="custom",
    api_mode="chat_completions",
    model="qwen3.5-2b-q8_0-registry.gguf",
    enabled_toolsets=["<your_toolset_here>"],
    max_iterations=8,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
)
```

The important Psionic-specific pieces are:

- `base_url`
- `provider="custom"`
- `api_mode="chat_completions"`
- `api_key="dummy"`
- `model` matching the basename of the GGUF Psionic is serving

The exact toolset names come from your Hermes checkout, not from Psionic.

### 3. Send Hermes a real task

Once Hermes is pointed at Psionic, you use Hermes the same basic way you would
use it against any other OpenAI-compatible backend:

- ask a plain question and get a text answer
- ask for a tool-backed task and let Hermes call tools
- let Hermes loop across tool calls until it has enough information

Examples of the kinds of tasks the retained lane already proves:

- "Use the weather tool for Paris."
- "Use both weather tools now, then summarize both cities in one answer."
- "Use the Atlantis weather tool and then explain truthfully what happened."

## What This Is Not Yet

This is not yet:

- a bundled Psionic-only Hermes product
- a one-command generic assistant launcher shipped by Psionic
- a claim that Psionic is faster than Ollama on every Hermes workload
- a claim that `llama.cpp` is already a clean comparator for this exact
  `qwen3.5` artifact lane

So the right mental model is:

- Psionic is now a working Hermes backend
- Hermes remains the agent/controller layer
- the integration is real and rerunnable
- the current remaining work is mostly benchmark and packaging polish, not
  tool-loop correctness

## Best Next Docs

If you want more than the quickstart:

- direct compatibility proof:
  `docs/HERMES_QWEN35_COMPATIBILITY.md`
- strict same-turn parallel proof:
  `docs/HERMES_QWEN35_PARALLEL_ATTRIBUTION.md`
- same-host Psionic vs Ollama benchmark:
  `docs/HERMES_BACKEND_BENCHMARK.md`
- serialized two-tool workflow:
  `docs/HERMES_QWEN35_SERIALIZED_TWO_CITY.md`
