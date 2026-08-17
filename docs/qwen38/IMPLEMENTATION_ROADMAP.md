# Qwen3.8 Implementation Roadmap

> Status: `partial` on 2026-08-17. Upstream research, artifact acquisition, and
> R1 product/artifact identity, R2 prompt/tokenizer contracts, R3 checkpoint
> admission, R4 bounded BF16 evidence, and R5 GGUF qualification are
> `implemented`; R6 native CPU generation is also `implemented` for the
> internal CPU lane. R7 native CUDA generation is `implemented`. R8
> OpenAI-compatible CPU/CUDA serving is `implemented_early`; R9 comparator and
> release gating is `implemented`. Later backend, multimodal, training, and
> performance milestones remain `planned`.

## Goal

Add a first-class Qwen3.8 model identity to Psionic, reuse the existing
`qwen3_5_text` architecture work where the artifact proves compatibility, and
ship one bounded native text-generation lane with deterministic evidence and
explicit refusal outside its accepted envelope.

The first claim targets `Qwen/Qwen3.8-27B` text generation from the admitted
`Qwen3.8-27B-UD-Q3_K_XL.gguf` artifact on CPU and CUDA. The official BF16
safetensors checkpoint remains the architecture and tensor-table authority.
The artifact decision and hardware envelope are recorded in
[FIRST_GGUF_TARGET.md](FIRST_GGUF_TARGET.md).

The upstream implementation inputs are pinned in
[UNSLOTH_CODE_AUDIT.md](UNSLOTH_CODE_AUDIT.md) and
[LLAMA_CPP_CODE_AUDIT.md](LLAMA_CPP_CODE_AUDIT.md). The llama.cpp comparator
revision is `9b05354ec6fb58b4e665e9a39ebc40285c015638`.

## First Claim Boundary

The first `implemented_early` claim includes:

- a distinct `qwen38` product-family identity
- official model, tokenizer, template, processor, and index digests
- exact Qwen3.8 chat-template behavior
- family-neutral `qwen3_5_text` architecture admission
- deterministic BF16 header, sampled-projection, and row-sparse full-layer
  evidence
- one qualified quantized GGUF artifact
- native CPU and CUDA text generation without a subprocess proxy
- truthful `/v1/models`, `/v1/chat/completions`, and `/v1/responses`
  publication
- thinking, non-thinking, tool-call, streaming, and response-state behavior
  inside an explicitly tested envelope
- explicit media refusal unless native vision execution has separately landed

The first claim excludes:

- native image or video understanding
- 1,000,000-token YaRN operation
- full BF16 CUDA residency on the local 16 GiB RTX 4080
- Qwen3.8 training, LoRA serving, DPO, GRPO, or adapter promotion
- automatic compatibility with every community GGUF conversion
- MTP acceleration, CUDA MTP, or multi-token draft batches
- native Metal execution or performance claims
- an Ollama or llama.cpp subprocess presented as Psionic execution

## Architecture Decision

Separate artifact identity from execution architecture:

| Concern | Identity |
| --- | --- |
| Product family | `qwen38` |
| Official model | `Qwen/Qwen3.8-27B` |
| Served model | `qwen3.8-27b` |
| Decoder architecture | `qwen3_5_text` |
| GGUF architecture | admitted from artifact metadata; expected `qwen35` |
| Prompt template | new digest-bound Qwen3.8 template version |
| Native context | 262,144 tokens, subject to runtime memory admission |
| Extended context | separate planned YaRN capability |

The reusable decoder contract should not remain under Qwen3.6-only type names.
Qwen3.6 and Qwen3.8 wrappers must preserve their own model ids, schema ids,
template ids, report text, artifact digests, and refusal reasons.

Do not copy the existing Qwen3.5 runtime into a second large model-local
module. Extract only the architecture-level pieces required to share the
hybrid decoder while keeping per-product admission and publication explicit.

## Milestone Summary

| Milestone | Status | Deliverable | Claim impact |
| --- | --- | --- | --- |
| R0 | `implemented` | Pinned upstream research and complete verified BF16 artifact | None; research only |
| R1 | `implemented` | Committed Qwen3.8 artifact-fact fixture and product identity | None; metadata only |
| R2 | `implemented` | Exact Qwen3.5 pretokenizer and Qwen3.8 prompt-template contract | None; frontend only |
| R3 | `implemented` | Family-neutral `qwen3_5_text` checkpoint admission | None; admission only |
| R4 | `implemented` | Real BF16 bounded execution evidence | Bounded evidence, not generation |
| R5 | `implemented` | Converter-bound GGUF, exact type support, and memory admission | Artifact admitted, not served |
| R6 | `implemented` | Native CPU token generation | Internal executable text lane with retained parity |
| R7 | `implemented` | Native CUDA token generation | First local accelerated lane |
| R8 | `implemented_early` | OpenAI-compatible serving and agent behavior | Candidate `implemented_early` claim |
| R9 | `implemented` | Comparator and correctness-first release gate | Retained `implemented_early` claim |
| R9A | `implemented` | Optional CPU MTP speculative decoding and rollback | Correctness implementation; no acceleration claim |
| R10 | `partial` | Native Metal generation | Runtime admitted; retained Apple evidence pending |
| R11 | `partial` | Native vision lane | Separate multimodal claim |
| R12 | `planned` | Training and adapter lane | Separate training claim |
| R13 | `planned` | Psionic exceeds the pinned Unsloth-equivalent speed-test | Bounded performance claim |

## Issue Tracking

[GitHub issue #1157](https://github.com/OpenAgentsInc/psionic/issues/1157)
tracks the complete roadmap. Each unfinished phase has its own scope,
dependency, evidence, validation, and merge-to-main close gate:

| Phase | Issue |
| --- | --- |
| R1 | [#1143: artifact facts and product identity](https://github.com/OpenAgentsInc/psionic/issues/1143) |
| R2 | [#1144: exact tokenizer and prompt contract](https://github.com/OpenAgentsInc/psionic/issues/1144) |
| R3 | [#1145: family-neutral checkpoint admission](https://github.com/OpenAgentsInc/psionic/issues/1145) |
| R4 | [#1146: official BF16 bounded evidence](https://github.com/OpenAgentsInc/psionic/issues/1146) |
| R5 | [#1147: GGUF qualification and storage support](https://github.com/OpenAgentsInc/psionic/issues/1147) |
| R6 | [#1148: native CPU generation](https://github.com/OpenAgentsInc/psionic/issues/1148) |
| R7 | [#1149: native CUDA generation](https://github.com/OpenAgentsInc/psionic/issues/1149) |
| R8 | [#1150: OpenAI-compatible serving](https://github.com/OpenAgentsInc/psionic/issues/1150) |
| R9 | [#1151: comparator and release gate](https://github.com/OpenAgentsInc/psionic/issues/1151) |
| R9A | [#1152: optional MTP decoding and rollback](https://github.com/OpenAgentsInc/psionic/issues/1152) |
| R10 | [#1153: native Metal generation](https://github.com/OpenAgentsInc/psionic/issues/1153) |
| R11 | [#1154: native image and video understanding](https://github.com/OpenAgentsInc/psionic/issues/1154) |
| R12 | [#1155: training, recovery, and adapters](https://github.com/OpenAgentsInc/psionic/issues/1155) |
| R13 | [#1156: beat the Unsloth-equivalent speed-test](https://github.com/OpenAgentsInc/psionic/issues/1156) |

## R0: Source Baseline

Current state:

- official revision:
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- complete artifact at `target/models/qwen/Qwen3.8-27B`
- 32 repository files verified by Hugging Face CLI
- 18 BF16 safetensors shards
- 1,199 indexed tensors
- official model facts and file roles documented in this directory

No implementation should silently refresh these inputs from upstream `main`.
An upstream revision change requires a reviewed digest update and replay of all
artifact, template, tensor, and parity gates.

## R1: Artifact Facts And Product Identity

Status: `implemented` in `fixtures/qwen38/qwen38_27b_artifact_facts_v1.json`
and `crates/psionic-models/src/qwen38.rs`. The stable artifact-facts and
admission schema versions are `psionic.qwen38.artifact_facts.v1` and
`psionic.qwen38.artifact_admission.v1`. This milestone adds metadata identity
and deterministic refusal only; it does not admit checkpoint execution.

Add a small committed fixture under `fixtures/qwen38/` without model weights.
The fixture should contain:

- upstream repository revision
- file digests from `MODEL_FACTS.md`
- normalized model and served ids
- architecture and processor facts
- tokenizer and template digests
- safetensors tensor count, tensor-data bytes, and shard inventory
- declared native and extended-context posture

Add Qwen3.8-specific Rust types in `psionic-models` for model identity,
artifact facts, and admission results. Do not label the artifact Qwen3.6 or
plain Qwen3.5.

Acceptance:

- official and explicit short model ids normalize deterministically
- unknown Qwen3.8 variants refuse instead of inheriting the 27B contract
- fixture drift identifies the exact changed field
- the official multimodal wrapper is recorded even when the active lane is
  text-only
- serialization is stable and digest-bound

Targeted validation:

```bash
cargo test -p psionic-models qwen38_artifact
```

## R2: Tokenizer And Prompt Contract

Status: `implemented` in `crates/psionic-models/src/qwen38_prompt.rs`, the
dedicated `qwen35` branch in `runtime_tokenizer.rs`, and
`fixtures/qwen38/qwen38_prompt_tokenizer_golden_v1.json`. Prompt receipts bind
the Qwen3.8 template and tokenizer digests plus effective reasoning,
preservation, tools, media-marker, and generation-frame settings. The GGUF
runtime uses the published per-code-point numeric split and NFC normalization
only for `qwen35`; other tokenizer families retain their existing patterns.
This frontend milestone does not admit model weights or generation.

The retained `llama-tokenize` comparison at revision
`9b05354ec6fb58b4e665e9a39ebc40285c015638` matches eight of nine tokenizer
cases against the selected GGUF. Its only mismatch is decomposed accented text:
the GGUF path does not apply the official NFC normalizer. Psionic follows the
official tokenizer for that case and records both token sequences in the
golden fixture. llama.cpp remains the regex-boundary comparator, not the
normalization authority.

Implement the published Qwen3.8 prompt semantics as a distinct template
version. Keep the upstream Jinja digest in the fixture and use parsed semantic
cases to validate the Rust renderer.

Required cases:

- empty-message refusal
- system, developer, user, assistant, and tool roles
- thinking enabled by default
- `reasoning_effort = xhigh`, `medium`, and `low`
- unsupported reasoning-effort refusal
- explicit non-thinking mode
- `preserve_thinking = true` by default
- disabled preserved thinking
- assistant `reasoning_content`
- single and multiple tool calls
- adjacent tool-result grouping
- image and video marker projection
- image/video refusal inside a system message
- generation-prompt framing for thinking and non-thinking modes

Tokenizer fixtures must prove actual Qwen3.8 token ids and rendered token
sequences. Matching vocabulary size and special ids with Qwen3.6 are not enough.

Implement a dedicated `qwen35` byte-level pretokenizer. It must preserve the
published use of Unicode combining marks and split each numeric code point
independently. Do not route `qwen35` through the current generic
`\p{N}{1,3}` digit grouping.

Required tokenizer edge cases:

- ASCII digit runs of lengths one through six
- non-ASCII numeric code points
- precomposed and decomposed accented words
- contractions in mixed case
- punctuation followed by newlines
- leading, trailing, and newline-adjacent whitespace

Acceptance:

- rendered bytes match golden upstream cases
- token ids match an upstream-supported reference tokenizer
- token ids match pinned llama.cpp for the dedicated Qwen3.5 regex-boundary
  cases; any normalizer difference is retained with both token sequences
- template and tokenizer digests are present in receipts
- prompt cache identity changes when reasoning or preservation settings change

Targeted validation:

```bash
cargo test -p psionic-models qwen38_prompt
cargo test -p psionic-models qwen38_tokenizer
```

## R3: Family-Neutral Checkpoint Admission

Status: `implemented` in `qwen35_text_checkpoint.rs`, the compatibility
wrappers in `qwen36_forward_admission.rs`, and
`qwen38_forward_admission.rs`. The Qwen3.8 report schema is
`psionic.qwen38_27b_forward_admission.v1`. The official checkpoint admits 851
decoder tensors and 15 MTP tensors across 18 index-derived shards. The other
333 indexed tensors remain separately inventoried. Eight decoder layers span
two shard files and publish explicit tensor-to-shard resolutions.

The shared parser rejects duplicate tensor keys before they can collapse into
a map. Header inspection also rejects a tensor observed in multiple shards.
Admission reports retain missing tensors, shape drift, dtype drift, index and
header disagreement, and index-to-observed shard drift separately. This phase
reads safetensors headers only. It does not read tensor data or execute the
checkpoint. The tensor-admission digest excludes local filesystem paths.

Extract the reusable dense `qwen3_5_text` tensor specification from
`qwen36_forward_admission.rs`. Preserve Qwen3.6 public wrappers while moving
architecture parsing, tensor-spec generation, safetensors header inspection,
and shared error details behind a family-neutral boundary.

Add Qwen3.8 wrappers with distinct schema versions and report text.

Acceptance:

- all expected Qwen3.8 text tensors admit from the official index
- 333 vision or other non-text tensors remain separately inventoried
- shard count is read from the index rather than hard-coded
- split layers resolve through the tensor-to-shard map
- missing tensor, dtype drift, shape drift, duplicate mapping, bad shard, and
  unsupported architecture cases refuse deterministically
- Qwen3.6 tests remain green with unchanged product identity

Targeted validation:

```bash
cargo test -p psionic-models qwen36_forward
cargo test -p psionic-models qwen38_forward
```

## R4: Official BF16 Bounded Evidence

Status: `implemented` through `qwen38_bf16_evidence.rs` and the
`qwen38_bf16_evidence` example. Three reviewed reports are retained under
`fixtures/qwen38/reports/`:

- `qwen38_bf16_header_admission_v1.json`
- `qwen38_bf16_sampled_projection_v1.json`
- `qwen38_bf16_bounded_traversal_v1.json`

The header report replays 18 shards and 1,199 mappings. The sampled report reads
one embedding row and three LM-head rows and retains deterministic sampled
logits. The traversal report reads one deterministic BF16 row from all 851
decoder and 15 MTP tensors, visits 64 decoder layers plus one MTP layer in
order, and retains 866 row receipts. All reports bind the config, tokenizer,
template, index, prompt-token, checkpoint-admission, tensor-read, and output
digests. Their capability object sets full-width attention, full-width MLP,
full-vocabulary logits, token generation, training gradients, and media
execution to `false`.

Add a Qwen3.8 example or command surface with three explicit backends:

- header admission
- sampled embedding/LM-head projection
- bounded row-sparse full-layer traversal

Run it against `target/models/qwen/Qwen3.8-27B`. This host has enough system
RAM for bounded BF16 validation, but the commands must retain the one-core
default where training infrastructure is involved.

Acceptance:

- all 18 shards and 1,199 tensor mappings are verified
- sampled real BF16 rows produce deterministic digests
- all 64 text layers and the declared MTP layer are visited in order
- reports bind config, tokenizer, template, index, tensor-read, and output
  digests
- reports explicitly deny full-width attention, full-vocabulary logits,
  generation, and training gradients

No `implemented_early` inference claim follows from R4.

## R5: GGUF Qualification Gate

Status: `implemented` through the GGUF loader/storage support and the retained
reports under `fixtures/qwen38/reports/`:

- `qwen38_gguf_converter_parity_v1.json`
- `qwen38_gguf_dynamic_v3_qualification_v1.json`
- `qwen38_gguf_q3_k_m_qualification_v1.json`
- `qwen38_gguf_q4_k_m_qualification_v1.json`

The primary Dynamic V3 artifact is admitted for native runtime implementation
and 4,096-token CUDA residency preflight. The 8,192-token estimate remains
unadmitted until R7 runtime peak memory and parity evidence exist. `Q3_K_M` is
retained as the standard K-quant comparison baseline. `Q4_K_M` is retained as a
CPU-offload quality comparator because full CUDA residency refuses on the
local RTX 4080 estimate. This phase does not generate tokens, serve requests,
or make Dynamic V3 canonical for output quality.

The primary candidate for the first local native generation lane is
`Qwen3.8-27B-UD-Q3_K_XL.gguf` from
`unsloth/Qwen3.8-27B-GGUF` at observed revision
`fdd03b8bbd279c1694563650e79d85a2373d9934`. Its published rounded size is
13.4 GB. The candidate is not required for R1-R4.

Use `Qwen3.8-27B-Q3_K_M.gguf` as the standard K-quant compatibility baseline
and `Qwen3.8-27B-Q4_K_M.gguf` as an explicit CPU-offload quality comparator.
Neither comparator replaces the primary artifact in the first CUDA residency
gate.

Record each candidate with:

- source repository and immutable revision
- exact filename, byte size, and SHA-256
- quantization mode and mixed-quantization inventory
- converter identity, revision, and source-model provenance when published or
  independently derived
- GGUF architecture and all family facts
- tensor inventory and dimensions
- tokenizer metadata and pre-tokenizer identity
- embedded chat-template digest
- native context and RoPE/MRoPE facts
- stored V-head layout and the evidence used to select it
- explicit MTP tensor disposition

Build the exact download plan before transfer. The plan must name every target
shard, expected byte size, expected digest, companion classification, and local
materialization path. The first text plan excludes `mmproj`.

The target filename is a profile label, not a runtime tensor type. R5 inspects
the actual GGUF tensor table and implements every required storage type for
the selected local set: `F32`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`,
`IQ3_S`, and `IQ4_XS`. Other community GGUFs still need independent tensor
inventory and support checks.

Validate the llama.cpp conversion contract with sampled official BF16 values:

- `A_log` is stored as negative exponential A
- time-step bias maps to the GGUF SSM time-step tensor
- non-linear-attention norm offsets are applied
- QKV, Z, alpha, beta, convolution, and output projection use tiled V-head
  order
- MRoPE sections are present and equal the admitted model contract

The official llama.cpp converter does not emit
`qwen35.ssm.v_head_reordered`. Record `v_head_layout = tiled` from converter
provenance and parity evidence. Unknown-producer artifacts without equivalent
evidence refuse.

Qualification checks:

- model dimensions match the official Qwen3.8 config
- tokenizer behavior matches the official tokenizer fixtures
- template behavior matches the Qwen3.8 prompt contract or is explicitly
  overridden by the digest-bound Psionic renderer
- tensor names and shapes map to the supported hybrid runtime
- sampled converter transforms match the official BF16 source
- every concrete GGML type maps to an implemented loader and runtime mode
- Q3_K block decode matches llama.cpp reference vectors when Q3_K is present
- no required tensor silently dequantizes or falls back to an unreported host
  path
- weights plus KV cache, recurrent state, scratch, graphs, and allocator margin
  fit the admitted hardware budget
- the 4,096-token CUDA envelope is proven before any longer-context claim
- the 8,192-token CUDA envelope has separate peak-memory and parity evidence
- output quality is compared with `Q3_K_M` before the Dynamic V3 artifact is
  made canonical
- MTP tensors are inventoried but skipped for standard generation

For the local 16 GiB RTX 4080, do not use file size alone as the admission
test. The chosen context window must leave explicit memory for KV cache,
recurrent state, scratch buffers, graph capture, and allocator overhead. A GGUF
that only fits its weights is not CUDA-admissible.

The first text lane excludes the approximately 930 MB BF16 and F16 vision
projectors. Native vision remains R11.

Use an operator path rather than a checked-in personal location:

```bash
PSIONIC_QWEN38_PILOT_GGUF_PATH=/absolute/path/model.gguf
```

Refuse GGUF artifacts whose metadata still identifies another product model
unless an explicit, reviewed compatibility receipt proves the conversion.

## R6: Native CPU Generation

Status: `implemented`. The selected Dynamic V3 GGUF enters a distinct `qwen38`
product family and executes through the native Psionic CPU Qwen3.5 hybrid
graph. Standard generation excludes the declared MTP tail and reports that
disposition. The lane remains internal; the generic OpenAI server refuses it
until R8.

Retained R6 evidence currently includes:

- a deterministic tiny fixture with three recurrent layers and one
  full-attention layer
- allocation checks proving F32 convolution/delta state exists only on
  recurrent layers and KV capacity exists only on the full-attention layer
- stable repeated requests with clean per-request state
- context-limit, host-memory, and stream-cancellation refusal checks
- a typed cooperative CPU generation timeout with stable `timed_out` and HTTP
  `504` diagnostics; zero-duration budgets refuse before execution and longer
  budgets are checked before and after each non-preemptible token step
- native greedy generation from `Qwen3.8-27B-UD-Q3_K_XL.gguf` without a
  subprocess or fallback
- exact raw-prompt first-token and two-token output parity with pinned
  llama.cpp revision `9b05354ec6fb58b4e665e9a39ebc40285c015638`
- retained layer-zero recurrent-intermediate parity for two-token prefill
  `[9419, 11]` and retained-state decode token `[353]`
- 28 passing tensor comparisons over RMS-normalized input, QKV projection,
  convolution, alpha/gate/beta, normalized Q/K, V, transposed delta state,
  recurrent output, gated normalization, and output projection
- maximum normalized RMSE `0.010121032189794241` and minimum cosine similarity
  `0.9999686621232524` across those comparisons

The retained report is
`fixtures/qwen38/reports/qwen38_cpu_recurrent_intermediate_parity_v1.json`.
The comparator source and pinned runner are
`scripts/qwen38-llama-cpp-intermediate-trace.cpp` and
`scripts/qwen38-llama-cpp-intermediate-trace.sh`. Reproduce and check it with:

```bash
scripts/qwen38-llama-cpp-intermediate-trace.sh \
  /home/christopherdavid/code/llama.cpp \
  target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  target/qwen38-llama-cpp-intermediate-trace
cargo run -p psionic-serve --example qwen38_cpu_intermediate_compare -- \
  target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  target/qwen38-llama-cpp-intermediate-trace \
  fixtures/qwen38/reports/qwen38_cpu_recurrent_intermediate_parity_v1.json
scripts/check-qwen38-cpu-intermediate-parity.sh
```

Generalize the existing Qwen3.5 hybrid runtime only as far as required for the
admitted Qwen3.8 GGUF. Validate these boundaries independently:

- embedding lookup
- RMS normalization
- Gated DeltaNet projections and recurrent state
- full-attention Q/K/V/output projections and output gate
- partial rotary embedding and interleaved MRoPE facts
- FFN gate/up/down projections
- residual ordering
- final normalization and LM head
- sampling and stop-token handling
- F32 convolution and delta-state allocation only on recurrent layers
- separate KV allocation only on full-attention layers
- MTP posture; standard generation skips and reports MTP tensors

Acceptance:

- deterministic tiny-fixture generation
- real-artifact greedy generation
- first-token and multi-token parity against a reference runtime
- stable repeated generation with clean per-request recurrent state
- prefill and token-at-a-time decode parity at recurrent intermediate boundaries
- no subprocess proxy and no unreported fallback
- cancellation, timeout, context limit, and memory refusal remain functional

Targeted validation:

```bash
cargo test -p psionic-serve qwen38_cpu
```

R6 can publish an executable CPU lane internally, but it does not complete the
first accelerated support claim.

## R7: Native CUDA Generation

Status: `implemented` on 2026-08-17. The shared native CUDA graph admits
Qwen3.8, preserves Qwen3.8-specific execution-plan and graph-cache namespaces,
passes the portable tiny-fixture acceptance suite, and has retained
production-artifact greedy and bounded-sampling rows from an idle RTX 4080.

The current implementation keeps the selected Dynamic V3 tensor storage on
CUDA without dense F16 mirrors. The `Q3_K` token embedding uses native
compressed row lookup. Mixed full-attention Q/K/V parts upload independently
when their quantization modes differ. Recurrent convolution, Gated DeltaNet,
full attention, FFN, final projection, greedy selection, bounded top-k
selection, and raw-logit materialization remain on the existing native CUDA
plan.

Qwen3.8 admission reads live free and total CUDA memory before the first
weight upload. It refuses unsupported quantization or insufficient total/free
memory before partial execution. The machine-readable runtime contract records
artifact, plan, graph, context, exact device-weight, recurrent-state, KV,
scratch, dense-mirror, raw-logit, and fallback truth. The admitted context is
4,096 tokens. CUDA allocator telemetry records current and peak Psionic-owned
device bytes, including retained pool buffers, so retained runs report a
measured allocation high-water mark in addition to the planned envelope. No
8,192-token claim follows from the current implementation.

The retained reports are
`fixtures/qwen38/reports/qwen38_cuda_greedy_generation_v1.json` and
`fixtures/qwen38/reports/qwen38_cuda_bounded_sample_generation_v1.json`. Both
bind the exact pretokenized prompt `[9419]`, the selected artifact byte length
and SHA-256, the idle check, and NVIDIA GeForce RTX 4080 GPU identity. Greedy
decode produced `[11, 353]` on both repeats at mean `10.995421174982495`
tokens/s. Seeded bounded sampling produced `[11, 271]` on both repeats at mean
`10.874144370111445` tokens/s. Both rows recorded zero host fallback, no raw
logits, graph hits with no shape drift, and the Qwen3.8 graph-cache identity.
The measured Psionic allocator high-water mark was `13,390,641,048` bytes,
below the `15,275,674,688` byte preflight requirement and within the
`15,808,397,312` bytes free at admission.

Reuse the existing Qwen3.5 CUDA kernels and execution plans only after CPU
parity proves the shared architecture boundary. Add Qwen3.8-specific plan and
artifact identities without cloning the entire runtime.

Acceptance:

- admitted weights and state stay inside the published residency envelope
- native Q3_K projections execute without a dense F16 mirror when Q3_K is
  present in a required projection
- SSM convolution and Gated DeltaNet run natively for both prefill and decode
- CPU and CUDA agree on bounded logits or token sequences under fixed inputs
- greedy and bounded sampled decode both work inside declared envelopes
- graph capture and replay report Qwen3.8-specific cache identity
- raw-logit materialization and host fallback are machine-visible
- out-of-memory and unsupported quantization refuse before partial execution
- repeated requests reset recurrent state and do not corrupt later outputs

Before every throughput run, verify the GPU is idle with the command required
by the repository agent contract:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits
```

Targeted validation:

```bash
cargo test -p psionic-serve qwen38_cuda
```

The retained evidence driver and checker are:

```bash
scripts/run-qwen38-cuda-generation-evidence.sh
scripts/check-qwen38-cuda-generation.sh
```

The driver verifies the selected artifact byte length and SHA-256, builds the
release benchmark, applies the required idle-GPU gate before each measured
process, and produces separate greedy and bounded-sampling reports. It refuses
without loading the model while any compute process is resident.

## R8: OpenAI-Compatible Serving

Status: `implemented_early` on 2026-08-17. The generic server admits Qwen3.8
on CPU and CUDA while preserving the external `qwen38` product identity.
Metal remains refused until R10.

Register the admitted Qwen3.8 runtime with the generic
`psionic-openai-server`. Publication must remain model-specific.

Required surfaces:

- `/health`
- `/v1/models`
- `/v1/chat/completions`
- streamed chat-completion deltas
- `/v1/responses`
- stored response-state replay across tool turns

Required behavior:

- Qwen3.8 model and artifact identity in headers and receipts
- effective thinking and reasoning-effort settings
- preserved-thinking behavior across conversation history
- tool choice `none`, `auto`, `required`, and named tools
- ordered and parallel tool calls
- streamed tool-call deltas
- structured-output support only inside the proven runtime envelope
- explicit raw-logit fallback truth where allowed
- image/video refusal on the text-only lane
- session reuse, adapters, prefix cache, and unsupported sampling controls
  published according to actual runtime support

Acceptance:

```bash
cargo test -p psionic-serve qwen38_openai
```

The focused acceptance suite covers health and model publication, chat and
streamed deltas, Responses storage and replay, reasoning controls, preserved
thinking, native Qwen3.8 XML tool calls, all tool-choice modes, ordered and
parallel calls, streamed tool arguments, structured output inside the proven
runtime envelope, CPU/CUDA capability rows, and text-only media refusal.
Responses and headers bind model key, served identity, artifact SHA-256,
backend, execution mode, prompt-template and tokenizer digests, prompt-cache
identity, effective thinking settings, and raw-logit posture. The lane has
therefore moved to `implemented_early`. R9 still owns the consolidated release
checker and comparator gate.

## R9: Comparator And Release Gate

Status: `implemented` on 2026-08-17. The retained report is
`fixtures/qwen38/reports/qwen38_release_gate_v1.json` and binds clean Psionic
revision `99283e19af6e851b79ac01480ec590c5e4d4764e`.

Add a release checker modeled on the existing Qwen3.5 pilot without reusing
Qwen3.5 names or claims.

The Qwen3.8-specific release inputs and commands are:

- `fixtures/qwen38/qwen38_release_template_cases_v1.json`
- `scripts/release/run-psionic-qwen38-release-gate.sh`
- `scripts/release/check-psionic-qwen38-release.sh`

The runner refuses unless Psionic is a clean `main` checkout equal to
`origin/main`, the selected Dynamic V3 artifact matches its pinned byte length
and SHA-256, and the CPU-only llama.cpp comparator checkout and binary both
match the pinned revision. The checker replays the portable test matrix and
validates the retained CPU, CUDA, fixture, source-revision, and comparator
digests. The generated report remains incomplete evidence until it is reviewed
and committed.

The retained run passes seven gates: artifact contract, prompt contract,
Qwen3.6 regressions, direct native generation and state reset, generic OpenAI
serving, the CPU recurrent comparator, and CUDA publication. The pinned
llama.cpp `/apply-template` endpoint matches Psionic byte-for-byte for explicit
`low`, `medium`, and `xhigh` `chat_template_kwargs`. Its 28 retained recurrent
intermediate comparisons also pass. CUDA publication records zero host
fallback, two graph hits, zero graph-shape drift, and a
`13,390,641,048`-byte allocator peak inside the preflight envelope.

The bundle retains the measured CUDA decode observations from R7 and the live
template-request latencies, but marks the performance claim as
`not_published`. R13 owns the bounded competitive performance claim.

The gate should run:

- committed artifact and template fixtures
- Qwen3.6 regression tests
- Qwen3.8 model and serving tests
- direct native generation
- generic-server generation
- tool-loop replay
- structured-output acceptance or refusal
- repeated-request state-reset checks
- comparator cases against one upstream-supported runtime
- comparator template cases with `low`, `medium`, and `xhigh` passed through
  `chat_template_kwargs`
- fallback-free CUDA publication checks

Retained reports must record:

- Psionic revision and dirty-tree posture
- model and artifact digests
- runtime and backend identity
- effective context and sampling settings
- prompt and expected-output contract
- correctness outcome
- latency and throughput without making an unsupported performance claim
- fallback, graph, cache, allocation, and refusal metrics

The initial release bar is correctness and truthful runtime publication.
Performance tuning follows with separate retained evidence.

Pin comparator reports to llama.cpp revision
`9b05354ec6fb58b4e665e9a39ebc40285c015638`. A comparator version change
requires prompt, tokenizer, intermediate, output, and performance evidence to
be regenerated.

## R9A: Optional MTP Speculative Decoding

MTP is not required for the first Qwen3.8 text claim. Add it only after the
base trunk is stable.

Status: `implemented` on 2026-08-17. The native CPU service exposes an
explicit `from_gguf_path_with_qwen38_mtp` constructor. The default constructor
still excludes the MTP tail and preserves the R6/R9 output path. The opt-in
constructor requires one declared NextN block and loads its 15 `blk.64.*`
tensors, while continuing to share the token embedding and LM head exactly as
the selected artifact declares.

The current implementation supports greedy decode with one draft per target
verification cycle. It passes the normalized target hidden row from the prior
target step together with the next-token embedding in llama.cpp order, and it
uses a separate dense-attention KV state for the MTP block. Target verification
clones the complete recurrent and full-attention state, advances through the
accepted token and proposed token, restores the snapshot after rejection, and
replays only the accepted prefix. Replay state, logits, and final hidden rows
must match before `restored_state_parity` remains true. An accepted draft also
runs one MTP alignment step from the verified target hidden row so the separate
MTP KV positions remain contiguous across later draft cycles.

`Qwen38MtpExecutionReport` uses schema
`psionic.qwen38.mtp_execution.v1`. It records draft, acceptance, rejection,
target-forward, replay, rollback, MTP-weight, MTP-KV, rollback-snapshot,
latency, and throughput facts. The producer and checker are
`scripts/run-qwen38-cpu-mtp-evidence.sh` and
`scripts/check-qwen38-cpu-mtp-evidence.sh`. Token-at-a-time target verification
is a correctness implementation. It does not claim acceleration. CUDA MTP,
sampling, structured output, and draft batches wider than one remain refused.

The retained selected-artifact report is
`fixtures/qwen38/reports/qwen38_cpu_mtp_evidence_v1.json`, generated from clean
revision `666f23fc6ae1c3430acae9deb20a2d3bd732ffc2`. Prompt tokens `[9419, 11]`
produce `[353, 2688]` with and without MTP. The one real draft is accepted, the
separate MTP KV cache peaks at `19,456` bytes, the target rollback snapshot
peaks at `157,159,936` bytes, and the `208,427,008` bytes of appended weights
bring added peak residency to `365,606,400` bytes. Baseline decode measures
`0.014146408447138228` tokens/s and MTP decode measures
`0.013500536740221532` tokens/s, a ratio of `0.9543437679372707`. The retained
outcome is `slowdown_observed`; no acceleration claim is published.

Required work:

- conditionally load the one appended NextN block
- allocate its separate dense-attention KV cache and weights
- accept target hidden rows and next-token embeddings with exact alignment
- implement bounded recurrent-state snapshots and rollback for rejected draft
  tokens
- report draft count, accepted count, acceptance rate, added residency, and
  rollback activity
- preserve base-model output parity when MTP is disabled

Acceptance requires correctness at draft boundaries, restored-state parity
after partial rejection, a measured memory envelope, and a retained
performance result. MTP presence in the artifact does not imply this claim.

## R10: Native Metal Generation

Status: `partial` on 2026-08-17. The family-neutral Metal service now admits
Qwen3.8 with a Qwen3.8-specific execution-plan namespace and a 4,096-token
context envelope. Admission enumerates every required output, attention, SSM,
and FFN projection before execution. Required projections without a native
Metal quantized kernel are refused; the Qwen3.8 lane does not use the existing
Qwen3.5 host-projection fallback. The qualified
`Qwen3.8-27B-Q4_K_M.gguf` storage types are inside the admitted native set.

The machine-readable Metal runtime contract reports artifact and plan
identity, device capacity and execution budget when available, device-visible
weight bytes, host recurrent-state and KV-cache bytes at the admitted context,
full layer and projection counts, conversion count, host-stepped state truth,
and host-projection fallback posture. Generic OpenAI serving now publishes
Qwen3.8 Metal as `backend = metal`, `execution_mode = native`,
`execution_engine = psionic`, and `fallback_policy = refuse`. Portable tests
cover load-plan publication and preflight refusal. Device-gated tests cover
bounded CPU/Metal logits, repeated request reset, runtime residency, and live
generic-server publication.

The serial retained-evidence driver is
`scripts/run-qwen38-metal-generation-evidence.sh`; its checker is
`scripts/check-qwen38-metal-generation.sh`. It runs one CPU row followed by two
Metal rows from a clean `origin/main` checkout after checking for competing
local model processes. R10 remains `partial` until that driver produces and
retains `fixtures/qwen38/reports/qwen38_metal_generation_evidence_v1.json` on
an idle Apple Silicon host. CUDA evidence does not fill this gap.

Reuse the family-neutral runtime and qualified GGUF after CPU correctness is
stable. Metal admission must keep quantization support, residency, cache, and
fallback truth separate from CUDA.

Acceptance:

- CPU and Metal agree on bounded logits or token sequences
- the full admitted layer stack has an explicit residency report
- unsupported quantized projections refuse or report an admitted conversion
- request state resets cleanly across repeated generations
- generic-server publication names `backend = metal`, native execution, and a
  refuse fallback policy
- full-model benchmark variants run serially on the interactive macOS host

Metal evidence does not inherit CUDA parity or performance results.

## R11: Native Vision

Status: `partial`. Psionic now has a separately admitted native vision source
artifact over the official first BF16 shard. Admission binds shard byte length
`3,966,730,552`, SHA-256
`ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c`,
all 333 `model.visual` tensors, `921,460,192` tensor bytes, and both processor
config digests. The loader reads only the vision prefix and reports the 59
non-vision tensors that share the source shard; no extracted or hidden mirror
artifact is required.

The bounded processor accepts decoded RGB8 media only when upstream smart
resize would preserve the dimensions. Image dimensions must be divisible by
32 and contain 65,536 through 262,144 pixels. The first video policy samples
at 2 fps, admits four through eight sampled frames, requires at most 65,536
pixels per frame, repeats the final frame for an odd temporal count, and caps
the combined patch count at 1,024. Inputs requiring resize refuse instead of
using a non-equivalent resampler. Receipts bind attachment bytes, frame
indices, dimensions, `grid_thw`, processor config, normalization, patch order,
limits, timeout, and the exact preprocessed tensor digest.

The native Candle-backed encoder executes patch projection, learned position
interpolation, rotary attention, all 27 vision blocks, spatial merge, and the
5,120-wide output projection on CPU or the feature-gated CUDA backend. Runtime
receipts publish all 333 resident tensors and 27 resident layers, backend,
engine, timeout, output identity, host materialization, `fallback_policy =
refuse`, and no hidden fallback. Tiny tests cover the complete graph and
layer-boundary timeout refusal.

The first full CUDA comparator driver is
`scripts/run-qwen38-vision-parity-evidence.sh`; its checker is
`scripts/check-qwen38-vision-parity.sh`. It pins Transformers revision
`0650ff354501cbdb7cb4138da628cc60f4e0ceed` and runs separate image and video
rows. The image row uses one 256x256 decoded RGB8 frame. The video row uses
eight 256x256 decoded RGB8 frames at 4 fps, verifies the upstream 2 fps sample
indices `[0, 2, 5, 7]`, and verifies the temporal grid `[2, 16, 16]`. Both rows
compare preprocessing and the complete 27-layer pooler output. The retained
rows are `fixtures/qwen38/reports/qwen38_vision_parity_v1.json` and
`fixtures/qwen38/reports/qwen38_vision_video_parity_v1.json`, both produced
from Psionic revision `4812efe2679e5dc68c0edd15b5d576d4f745c3f2` on an idle
RTX 4080 with driver `595.58.03`. Preprocessing and sampling are byte-exact.
The image encoder output has normalized RMSE `0.0667980101638391`, p99
absolute error `0.1171875`, and cosine similarity `0.9977859258651733`. The
video encoder output has normalized RMSE `0.05989835909389969`, p99 absolute
error `0.1328125`, and cosine similarity `0.9983055591583252`. Both compare
against pinned Transformers CUDA eager attention. The one-shot elapsed fields
are diagnostic and do not establish a performance claim.

The model layer now also constructs a strict multimodal decoder-input plan.
It expands each image marker to `grid_thw.prod() / 4` image pad tokens. Video
markers expand to upstream-compatible one-decimal timestamps and one
vision-delimited pad span per temporal grid. The plan validates full native
encoder receipts and materialized output digests, binds every 5,120-wide row
to an exact pad-token index, derives text/image/video token types, and computes
the Qwen3.5 three-axis MRoPE positions plus generated-token position delta.
Malformed dimensions, media ordering, token counts, output widths, runtime
fallbacks, and output digests refuse before decoder execution.

The native CPU decoder now consumes the plan through
`generate_qwen38_multimodal`. During prefill it replaces each exact pad-token
embedding with the admitted 5,120-wide encoder row and passes the planned
three-axis coordinate through every full-attention RoPE application. Recurrent
layers preserve their position-independent state transition. Generated tokens
use the physical KV-cache position plus the retained MRoPE delta. Text-only
calls preserve scalar-equivalent `[position, position, position]` behavior.
The CPU lane refuses context-window prompt truncation and MTP speculative
decode for multimodal calls because either would invalidate the admitted plan.
The service retains the successful plan receipt for inspection.

The native CUDA decoder consumes the same plan through
`generate_qwen38_multimodal`. Admitted 5,120-wide rows upload directly into
the resident decoder hidden buffer before layer zero. Multimodal prefill and
the position-delta decode steps use an uncaptured native CUDA path because the
existing text graph binds scalar RoPE decode parameters. Full-attention layers
run a fused F16-KV attention kernel with three positions, the GGUF MRoPE
sections, and the GGUF interleaving flag. Recurrent layers remain unchanged.
Text-only generation retains its captured graph path. Multimodal calls bypass
the token-only shared-prefix cache, refuse context-window prompt truncation,
and retain the successful plan receipt. The bounded
`qwen38_multimodal_cuda_smoke` example runs vision and decoder residency
serially so the two full models do not overlap in VRAM.

R11 remains `partial` until Metal consumes the plan and chat/responses media
serving passes attachment, streaming, tool, bound, malformed-input, and refusal
coverage with retained end-to-end generation evidence.

Native vision remains a separate roadmap lane after text support.

It requires:

- exact image and video preprocessing
- the 27-layer vision encoder
- spatial and temporal patch handling
- vision-to-text projection into the 5,120-wide decoder
- attachment identity and preprocessing receipts
- image and video reference fixtures
- native output parity against an upstream-supported runtime
- bounded image/video size, frame count, timeout, and memory admission

Until R11 lands, prompt marker projection must not be described as image or
video understanding.

## R12: Training And Adapters

Training remains separate from inference support. Reuse the Qwen legal and
open-adapter substrate only after native base-model execution is stable.

Required follow-on work includes:

- Qwen3.8-specific adapter identity
- exact backward kernels for admitted targets
- adapter serving on the same runtime
- checkpoint and optimizer-state recovery
- corpus and evaluation lineage
- promotion receipts tied to the Qwen3.8 base digest

Do not relabel Qwen3.6 or Qwen3.5 adapters as Qwen3.8-compatible.

## R13: Psionic Versus Unsloth-Equivalent Speed Gate

The final local performance target compares native Psionic with the equivalent
Unsloth GGUF path on the admitted RTX 4080 lane. The Unsloth Studio GGUF path
audited for this roadmap delegates execution to llama.cpp or `llama-server`;
the comparator report must therefore pin both the Unsloth revision and the
effective llama.cpp revision and build.

Use the same admitted GGUF digest, prompt token ids, generated-token count,
context, batch and microbatch sizes, sampling, stop conditions, cache policy,
warmup policy, GPU residency, and correctness contract. Record prefill
throughput, decode throughput, time to first token, end-to-end latency, peak
GPU and host memory, graph and cache behavior, and every fallback. Validate
output correctness before accepting a performance row.

The primary winning metric is median decode tokens per second. Psionic must
strictly exceed the Unsloth-equivalent median across retained repeated runs
without weakening R9 correctness, execution identity, state reset, memory, or
refusal guarantees. MTP results are separate unless both sides use an
equivalent speculative configuration.

Before each measured GPU process, run the idle-process query required by the
repository agent contract. Retain raw samples, summaries, exact replay
commands, source and artifact revisions, hardware and runtime facts, output
tokens, memory metrics, and the Psionic commit that produced the winning row.

## Test Matrix

| Layer | Synthetic fixture | Official BF16 | Qualified GGUF | CPU | CUDA | Metal |
| --- | --- | --- | --- | --- | --- | --- |
| Product identity | required | required | required | n/a | n/a | n/a |
| Tokenizer/template | required | required | required | n/a | n/a | n/a |
| Converter layout | required | sampled source | required | n/a | n/a | n/a |
| Quantized type decode | required | no | required | required | required | planned |
| Tensor admission | required | required | required | n/a | n/a | n/a |
| Bounded row evidence | required | required | optional | yes | no | no |
| Full token generation | required | comparator | required | required | required | planned |
| Tool and response replay | required | no | required | required | required | planned |
| Structured output | required | no | required | required | required or refused | planned or refused |
| Media | marker/refusal | processor facts | marker/refusal | refused | refused | refused |
| Performance | no | no | required | informational | retained | follow-on retained |
| MTP and rollback | accept/reject/partial plus tiny graph | bounded source | optional follow-on | implemented CPU | planned | planned |

Synthetic fixtures cover deterministic failures and small numerics. They do
not replace real-artifact tests. Full-model tests should use environment-gated
paths so ordinary unit tests remain portable.

## Commit Sequence

Land the implementation as narrow commits on `main`:

1. Qwen3.8 artifact fixture and product identity.
2. Tokenizer and template renderer with golden cases.
3. Family-neutral `qwen3_5_text` admission refactor with Qwen3.6 regression
   coverage.
4. Qwen3.8 BF16 header and bounded execution reports.
5. Qualified GGUF descriptor, converter-layout fixtures, exact quantized type
   support, memory plan, and refusal tests.
6. Native CPU generation and reference parity.
7. Native CUDA generation and backend-truth receipts.
8. OpenAI-compatible serving, tools, streaming, and response replay.
9. Release checker, comparator bundle, and canonical doc status update.
10. Optional MTP speculative decoding and rollback evidence.
11. Native Metal generation and Apple backend evidence.
12. Retained Psionic-versus-Unsloth-equivalent speed-test evidence and any
    focused optimizations required to exceed it.

Generated fixtures and reports land with the code that produces them. Do not
mix broad formatting, unrelated Qwen3.5 benchmark files, or incidental report
regeneration into these commits.

## External Inputs And Blockers

R6 can begin from the retained R0-R5 artifact, tokenizer, checkpoint,
converter, storage, and memory-preflight evidence.

R6 and later need:

- reference outputs from pinned llama.cpp revision
  `9b05354ec6fb58b4e665e9a39ebc40285c015638` for tokenizer, prompt,
  intermediates, logits, generation, and eventually vision
- prefill, decode, cache-reset, timeout, cancellation, and memory-refusal
  evidence for the admitted GGUF
- runtime-measured local GPU margin for the chosen artifact and context window,
  or a larger remote GPU for full-BF16 CUDA validation

These inputs do not block the template, admission, or bounded BF16 milestones.
They block honest native generation and accelerated release claims.

## Completion Rules

Move the lane to `implemented_early` only when R0-R8 are complete and the
retained release matrix has no hidden fallback or identity ambiguity.

Move the text lane to `implemented` only after:

- the release gate is stable across the retained hardware rows
- full supported sampling and cache behavior is documented
- repeated-load and long-run state tests are green
- comparator correctness is retained
- performance and memory envelopes are measured
- unsupported vision, extended-context, adapter, and training surfaces still
  refuse explicitly

Native vision and training keep their own statuses. Their absence must not
reduce the honesty of a completed text-only claim.
