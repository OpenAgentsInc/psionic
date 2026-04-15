# Gemma 4 26B Metal Device-Resident Decode Audit

Scope: close `#945` by pushing the local single-node `gemma4:26b` Metal decode
loop closer to a device-resident steady state, prove the new readback boundary
explicitly, and record the measured throughput delta on the retained benchmark
prompt.

## What Landed

This pass fixes a different class of problem than the earlier `Q5_0` FFN
enablement.

The earlier `#944` pass removed the local sparse FFN fallback cliff and moved
the retained single-node receipt from about `5.14 tok/s` to about `24.22 tok/s`
on the real `gemma-4-26B-A4B-it-Q4_K_M.gguf` artifact.

That still left too much host-facing work in the decode loop.

This pass lands four concrete changes:

- the Metal backend now has a raw single-row top-k decode kernel and host
  plumbing for bounded candidate readback instead of forcing dense host logits
  materialization for that path
- the Gemma 4 Metal decode loop now tracks explicit output-mode and readback
  receipts, so each request can report whether it used greedy token selection,
  bounded top-k candidates, or dense raw logits
- the normal single-node Metal path now avoids prompt-side host KV
  materialization unless prompt-cache recording or explicit staged behavior
  actually requires it
- the benchmark harness now records Gemma 4 Metal readback bytes per token and
  host KV materialization counts directly in the JSON receipt

The owning code is in:

- `crates/psionic-backend-metal/src/lib.rs`
- `crates/psionic-serve/src/gguf.rs`
- `crates/psionic-serve/src/lib.rs`
- `crates/psionic-serve/examples/gemma4_bench.rs`

## Acceptance Mapping

This issue asked for three things that are now visible and testable.

Greedy single-node decode:

- the hot path reads back only the selected token id
- the retained real-artifact benchmark now reports `4 B/token` of readback,
  which is exactly one `u32` token id per generated token
- the same benchmark reports `0` host KV materialization events on the timed
  request

Bounded top-k single-node decode:

- the native Metal path now has an explicit bounded top-k candidate mode
- the bounded path reads back only token ids and candidate values, not dense
  raw logits
- the coverage test proves that `top_k = 2` reports
  `2 * (sizeof(u32) + sizeof(f32))` bytes of readback and `0` host KV
  materialization events

Benchmark visibility:

- `gemma4_bench` now publishes:
  - `gemma4_metal_decode_output_modes`
  - `gemma4_metal_decode_readback_bytes`
  - `gemma4_metal_decode_readback_bytes_per_token`
  - `gemma4_metal_host_kv_materialization_events`
  - `gemma4_metal_host_kv_materialization_tokens`

## Validation

Targeted tests:

```bash
cargo test -p psionic-backend-metal \
  metal_backend_single_row_top_k_selection_returns_raw_logits_on_supported_hardware \
  -- --nocapture

cargo test -p psionic-serve metal_gemma4_service_reports_ -- --nocapture
```

Results:

- both passed
- the backend test proves the single-row Metal top-k path returns raw logits
  for the selected candidates instead of MoE-router-style softmax weights
- the service tests prove:
  - greedy readback is one token id per step
  - bounded top-k readback is a bounded candidate payload
  - the common synthetic fast path does not materialize host KV buffers

## Benchmark Receipt

Environment:

- host: `Apple M5 Max`
- memory: `128 GB`
- backend: `Metal`
- artifact:
  `/Users/christopherdavid/models/gemma4/all/gemma-4-26B-A4B-it-Q4_K_M.gguf`
- prompt mode: `rendered_tokens`
- output cap: `64` tokens

Command:

```bash
cargo run --release -q -p psionic-serve --example gemma4_bench -- \
  --model-path /Users/christopherdavid/models/gemma4/all/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --mode single \
  --backend metal \
  --prompt-mode rendered-tokens \
  --prompt "You are reviewing an inference engine change for single-node sparse Gemma 4 execution. Explain the tradeoffs between single-node sparse execution and distributed sparse execution for a 26B A4B Gemma model. Focus on latency, memory pressure, expert routing overhead, and operational simplicity.

Then give a short recommendation for when a local single-node Metal path is good enough for a developer workflow, and when a distributed path is still necessary.

Keep the answer concrete and technical. Use plain language and avoid marketing phrasing." \
  --max-output-tokens 64 \
  --repeats 1 \
  --json-out /tmp/psionic-gemma4-26b-issue945.json
```

Receipt:

- `load_s = 1.143`
- `prompt_s = 5.989`
- `decode_s = 2.163`
- `ttft_s = 6.023`
- `decode_tok_s = 29.58`
- `gemma4_metal_decode_output_modes = ["greedy_token"]`
- `gemma4_metal_decode_readback_bytes = 256`
- `gemma4_metal_decode_readback_bytes_per_token = 4.0`
- `gemma4_metal_host_kv_materialization_events = 0`
- `gemma4_metal_host_kv_materialization_tokens = 0`

Comparison against the earlier retained single-node receipt from `#944`:

- previous `decode_tok_s = 24.22`
- current `decode_tok_s = 29.58`
- throughput gain = `1.221x`
- percent gain = `22.13%`

What that means:

- the earlier FFN enablement removed the largest fallback cliff
- this follow-on pass removed more of the remaining host-facing decode tax
- the retained greedy lane is now measurably faster and its readback boundary is
  explicit instead of inferred

## Remaining Boundary

This issue closes the requested device-resident decode-state work for the
common greedy and bounded top-k paths. It does not close the full competitive
gap yet.

The current honest boundary is still:

- the local sparse 26B lane remains well behind same-host `ollama` and
  `llama.cpp`
- the shared benchmark prompt still produces malformed text on the local Metal
  sparse lane
- the remaining gap is no longer "full logits readback every step" or "host KV
  append every step" on the common path; the remaining work is deeper decode
  architecture, dispatch shape, and kernel efficiency
