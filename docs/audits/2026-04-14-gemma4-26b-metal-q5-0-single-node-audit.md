# Gemma 4 26B Metal Q5_0 Single-Node Audit

Scope: document the `#944` single-node Metal pass for sparse `gemma4:26b`,
freeze the benchmark delta on the real local GGUF artifact, and record the
remaining boundary honestly.

## What Landed

This change closes one specific architectural hole in the local Metal
single-node path for sparse `gemma4:26b`.

Before this pass:

- the sparse fused gate/up expert tensors loaded on the native Metal path for
  `GGML Q4_K`
- but `16` decoder layers still failed out of the fully local FFN path because
  their dense and sparse down-projection tensors were `GGML Q5_0`
- the Metal load plan therefore admitted only `14 / 30` native FFN layers on
  the real `gemma-4-26B-A4B-it-Q4_K_M.gguf` artifact
- the benchmarked local decode path stayed near the earlier `5 tok/s` range

After this pass:

- fused sparse `gate_up` expert tensors remain supported on the native Metal
  path for `GGML Q4_K`
- native Metal quantized projection support now includes `GGML Q5_0`
- native Metal sparse expert matvec support now includes `GGML Q5_0`
- the Metal backend now exposes:
  - `quantized_matvec_q5_0`
  - `quantized_matvec_argmax_q5_0`
  - `mul_mv_id_q5_0`
  - `expert_matvec_f32_ids_q5_0`
- the real local load plan now admits `30 / 30` native FFN layers on the same
  sparse Gemma 4 26B artifact

The owning code is in:

- `crates/psionic-backend-metal/src/lib.rs`
- `crates/psionic-serve/src/gguf.rs`

## Benchmark Receipt

Environment:

- host: `Apple M5 Max`
- memory: `128 GB`
- backend: `Metal`
- artifact:
  `/Users/christopherdavid/models/gemma4/all/gemma-4-26B-A4B-it-Q4_K_M.gguf`
- prompt mode: `rendered_tokens`
- output cap: `64` tokens

Prompt:

```text
You are reviewing an inference engine change for single-node sparse Gemma 4 execution. Explain the tradeoffs between single-node sparse execution and distributed sparse execution for a 26B A4B Gemma model. Focus on latency, memory pressure, expert routing overhead, and operational simplicity.

Then give a short recommendation for when a local single-node Metal path is good enough for a developer workflow, and when a distributed path is still necessary.

Keep the answer concrete and technical. Use plain language and avoid marketing phrasing.
```

Debug load proof on the updated path:

- `native_attention_layers = 30`
- `native_ffn_layers = 30`

Updated local receipt:

- `load_s = 1.189`
- `prompt_s = 6.140`
- `decode_s = 2.642`
- `ttft_s = 6.190`
- `decode_tok_s = 24.22`

Previous retained comparison point for the same host and artifact:

- `decode_tok_s = 5.14`

Measured improvement from this pass:

- `24.22 / 5.14 = 4.71x`

This closes the earlier local-FFN fallback cliff. It does not close the full
throughput gap against `ollama` or `llama.cpp`.

## Correctness Boundary

The visible text output on this sparse 26B lane is still malformed on the
shared benchmark prompt.

That problem is real, but this pass did not introduce it.

The reason for that claim is concrete:

- the native path now produces malformed text at about `24.22 tok/s`
- forcing host projections with
  `PSIONIC_GEMMA4_METAL_FORCE_HOST_PROJECTIONS=1` reproduces the same malformed
  prefix while collapsing decode throughput to about `0.68 tok/s`

That isolates the new `Q5_0` projection enablement away from the malformed text
symptom. The malformed output remains somewhere deeper in the existing sparse
Gemma 4 local lane and should be treated as a separate follow-on problem.

## Remaining Gap

This pass makes the local Metal sparse FFN path honest and materially faster.
It does not yet make Psionic competitive with the same-host `ollama` or
`llama.cpp` baselines for this artifact.

The remaining performance work is still architectural:

- keep decode state on device end-to-end
- bound readback to only the data required for token selection and receipts
- add an explicit benchmark gate that fails closed when the local lane falls
  below the target throughput envelope

Those items remain separate follow-on work after `#944`.
