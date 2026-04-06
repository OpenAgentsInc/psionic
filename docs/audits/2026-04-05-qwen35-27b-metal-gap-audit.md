# Qwen3.5 27B Metal Gap Audit

`Qwen3.5-27B-Q4_K_M.gguf` now runs natively in Psionic on local Metal. The hard
parsing and family-admission gaps are closed. The model now produces the same
early decode shape as the local Ollama lane on the raw prompt `who are you`:
`?\n\n<think>\nOkay, the user is asking "who are you?" I`.

The problem is no longer "Psionic cannot run the public 27B artifact." The
problem is that Psionic still runs it with the wrong execution shape on Apple
Silicon.

## Current Receipts

The most useful comparison is the exact same raw prompt on the exact same
artifact, before and after the latest Psionic Metal runtime fixes, against the
local Ollama reference lane.

| Runtime | Artifact | Load | Prompt | Decode | Total | TTFT | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Psionic Metal, before output-selection fix | `Qwen3.5-27B-Q4_K_M.gguf` | `1.916 s` | `0.748 s` | `2.689 s` | `3.436 s` | `0.193 ms` | `5.95` |
| Psionic Metal, current | `Qwen3.5-27B-Q4_K_M.gguf` | `1.833 s` | `0.539 s` | `2.305 s` | `2.844 s` | `0.050 ms` | `6.94` |
| Ollama local | `qwen35-27b-local:latest` | `3.234 s` | `0.333 s` | `0.785 s` | `4.384 s` | not published separately | `20.38` |

The current Psionic pass is materially better than the earlier one. Decode
throughput is up from `5.95 tok/s` to `6.94 tok/s`, about a `16.7%` gain on the
same prompt and host. That is real improvement. It is also still roughly `3x`
slower than the local Ollama lane on decode.

That remaining gap is large enough that it cannot be explained by one bad
constant, one wrong tensor fact, or one missing tokenizer rule. It is a runtime
shape gap.

## What We Fixed Already

The first class of bugs was correctness. Psionic now handles the public 27B
artifact much more honestly than the first bring-up did.

The native runtime now accepts the real artifact metadata that mattered for this
family: the scalar `qwen35.attention.head_count_kv` form, `blk.N.ssm_dt.bias`,
the official `qwen35` tokenizer pre, and the missing
`qwen35.ssm.v_head_reordered` family fact by defaulting it to `true`, which is
what the local Ollama reference expects for this family. The runtime also now
uses the GGUF MRoPE metadata instead of silently treating the path as generic
RoPE.

The second class of bugs was math. Two of them mattered directly. The hybrid
recurrent path was doing extra fake q/k normalization before the real
`delta_net_autoregressive_step_in_place` normalization, and the host hybrid path
was exponentiating the decay twice. Removing those mistakes moved the model from
prompt-suffix echo and garbage to coherent early decode.

The third class of bugs was serving architecture. Before this pass, the Metal
service still computed output projection for every prompt token and materialized
full-vocabulary logits even for greedy decode. The current Metal lane now skips
prompt-prefix output work entirely and keeps greedy decode on an argmax-only
path instead of reading back raw logits every step.

That is why the current lane is faster even though the core layer loop is still
the same shape.

## What Ollama And `llama.cpp` Are Doing Differently

The local references are not faster because they happen to parse one tensor
better. They are faster because `qwen35` is treated as a first-class hybrid
family in the runtime.

`llama.cpp` has dedicated `qwen35` and `qwen35moe` model builders, recurrent
memory ownership, and explicit fused gated-delta-net chunked paths. The
important local files are:

- `competition/repos/llama.cpp/src/models/qwen35.cpp`
- `competition/repos/llama.cpp/src/models/qwen35moe.cpp`
- `competition/repos/llama.cpp/src/llama-memory-recurrent.h`
- `competition/repos/llama.cpp/src/llama-memory-hybrid-iswa.cpp`

Ollama inherits that runtime shape in its vendored `llama.cpp` layer and also
owns the Qwen3-next conversion and model metadata path directly. The important
local files are:

- `competition/repos/ollama/model/models/qwen3next/model.go`
- `competition/repos/ollama/model/models/qwen3next/deltanet.go`
- `competition/repos/ollama/llama/llama.cpp/src/models/qwen35.cpp`

The local `mlx` checkout in this workspace does not currently expose a checked-in
`qwen35` implementation to compare line by line. MLX is still useful as the
Apple reference for the right execution philosophy: device-owned arrays,
backend-owned state, and unified-memory execution that does not bounce every
major activation back through host `Vec<f32>` control flow.

That is the real architectural line here. The reference systems keep Qwen35
inside a backend-owned execution path. Psionic still leaves too much of the
Metal Qwen35 path in host-owned Rust vectors and host-managed recurrent state.

## Where Psionic Still Loses Time

The current Metal lane is only partially accelerated.

Large quantized projections are already admitted natively on Metal. `Q4_K`,
`Q5_K`, `Q6_K`, `Q8_0`, and `MXFP4` are all admitted for the Qwen35 projection
path now. That part is no longer the blocker.

The blocker is that the service still steps the model on the host between those
projections. `MetalGgufQwen35TextGenerationService` still calls a
host-oriented `MetalQwen35Model::forward_token` that decodes token embedding to
a host vector, threads `Vec<f32>` hidden state through every layer, and reuses
CPU-shaped recurrent and attention state containers for the hybrid path. The
projection kernels are native. The execution ownership is not.

In practical terms, the current lane still pays for:

- host-visible hidden-state materialization between major substeps
- host-managed recurrent update flow instead of backend-owned recurrent state
- host-managed full-attention bookkeeping instead of a backend-native cache path
- one-token-at-a-time prompt replay instead of chunked recurrent prefill

That is why fixing output selection helps, but does not close the gap. We made
the last stage cheaper. The middle of the pipeline is still shaped wrong.

## What Has To Change Next

The next step is not "distribute it first." Distributing a host-stepped local
lane just spreads a slow execution shape across more machines.

The next step is to make Qwen35 on Metal a real backend-owned path.

That means four concrete changes.

First, the Metal Qwen35 lane needs a real step plan, the same way the CUDA lane
already has one. Hidden state, projection scratch, recurrent scratch, and output
selection buffers need to stay in backend-owned buffers instead of returning to
host `Vec<f32>` after every major operation.

Second, the hybrid recurrent state must stop aliasing CPU state types. The
current code literally reuses CPU hybrid and full-attention state containers in
the Metal lane. That is acceptable for bring-up and unacceptable for performance.

Third, prompt replay has to become chunked recurrent prefill instead of scalar
token stepping. `llama.cpp` is explicit about this in its hybrid memory and
chunked gated-delta-net path. Psionic needs the same class of shape on Metal if
it wants to be in the same performance conversation.

Fourth, once the local single-node Metal lane is backend-owned, then clustered
or split execution becomes meaningful. If we later decide to distribute Qwen35
for larger home-network or cross-site runs, the split should happen above a
device-owned local stage, not above the current host-driven layer loop.

## Bottom Line

Psionic no longer has a Qwen35 admission problem on Metal. It has a runtime
shape problem.

The current work closed the parser and math mistakes, made the model coherent,
and removed an obvious serving inefficiency by skipping pointless prompt logits
and keeping greedy decode on a device argmax path. That raised the native Metal
lane from `5.95 tok/s` to `6.94 tok/s` on the benchmark that matters here.

The remaining gap to Ollama is still large because Ollama and `llama.cpp` keep
Qwen35 inside a first-class backend-owned hybrid runtime, while Psionic still
hands too much of the Metal lane back to the host between steps.

The right next architecture change is therefore clear: make Qwen35 on Metal a
real backend-owned execution path first, then talk about distribution.
