# Qwen3.8 Implementation Roadmap

> Status: `planned` on 2026-08-14. Upstream research and artifact acquisition
> are `implemented`; Qwen3.8 execution in Psionic remains `planned`.

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
| R1 | `planned` | Committed Qwen3.8 artifact-fact fixture and product identity | None; metadata only |
| R2 | `planned` | Qwen3.8 tokenizer and prompt-template contract | None; frontend only |
| R3 | `planned` | Family-neutral `qwen3_5_text` checkpoint admission | None; admission only |
| R4 | `planned` | Real BF16 bounded execution evidence | Bounded evidence, not generation |
| R5 | `planned` | Qualified small GGUF and memory admission | Artifact admitted, not served |
| R6 | `planned` | Native CPU token generation | First executable text lane |
| R7 | `planned` | Native CUDA token generation | First local accelerated lane |
| R8 | `planned` | OpenAI-compatible serving and agent behavior | Candidate `implemented_early` claim |
| R9 | `planned` | Comparator, performance, and release gate | Retained `implemented_early` claim |
| R10 | `planned` | Native Metal generation | Separate Apple backend claim |
| R11 | `planned` | Native vision lane | Separate multimodal claim |
| R12 | `planned` | Training and adapter lane | Separate training claim |

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

Acceptance:

- rendered bytes match golden upstream cases
- token ids match an upstream-supported reference tokenizer
- template and tokenizer digests are present in receipts
- prompt cache identity changes when reasoning or preservation settings change

Targeted validation:

```bash
cargo test -p psionic-models qwen38_prompt
cargo test -p psionic-models qwen38_tokenizer
```

## R3: Family-Neutral Checkpoint Admission

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
- converter and source-model provenance when published
- GGUF architecture and all family facts
- tensor inventory and dimensions
- tokenizer metadata and pre-tokenizer identity
- embedded chat-template digest
- native context and RoPE/MRoPE facts

Qualification checks:

- model dimensions match the official Qwen3.8 config
- tokenizer behavior matches the official tokenizer fixtures
- template behavior matches the Qwen3.8 prompt contract or is explicitly
  overridden by the digest-bound Psionic renderer
- tensor names and shapes map to the supported hybrid runtime
- no required tensor silently dequantizes or falls back to an unreported host
  path
- weights plus KV cache, recurrent state, scratch, graphs, and allocator margin
  fit the admitted hardware budget
- the 4,096-token CUDA envelope is proven before any longer-context claim
- the 8,192-token CUDA envelope has separate peak-memory and parity evidence
- output quality is compared with `Q3_K_M` before the Dynamic V3 artifact is
  made canonical

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
- MTP posture; standard generation must state whether MTP tensors are unused

Acceptance:

- deterministic tiny-fixture generation
- real-artifact greedy generation
- first-token and multi-token parity against a reference runtime
- stable repeated generation with clean per-request recurrent state
- no subprocess proxy and no unreported fallback
- cancellation, timeout, context limit, and memory refusal remain functional

Targeted validation:

```bash
cargo test -p psionic-serve qwen38_cpu
```

R6 can publish an executable CPU lane internally, but it does not complete the
first accelerated support claim.

## R7: Native CUDA Generation

Reuse the existing Qwen3.5 CUDA kernels and execution plans only after CPU
parity proves the shared architecture boundary. Add Qwen3.8-specific plan and
artifact identities without cloning the entire runtime.

Acceptance:

- admitted weights and state stay inside the published residency envelope
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

## R8: OpenAI-Compatible Serving

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

R8 is the earliest point at which the lane may move from `planned` to
`implemented_early`, provided R0-R8 evidence is retained and the full claim
matrix is green.

## R9: Comparator And Release Gate

Add a release checker modeled on the existing Qwen3.5 pilot without reusing
Qwen3.5 names or claims.

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

## R10: Native Metal Generation

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

## Test Matrix

| Layer | Synthetic fixture | Official BF16 | Qualified GGUF | CPU | CUDA | Metal |
| --- | --- | --- | --- | --- | --- | --- |
| Product identity | required | required | required | n/a | n/a | n/a |
| Tokenizer/template | required | required | required | n/a | n/a | n/a |
| Tensor admission | required | required | required | n/a | n/a | n/a |
| Bounded row evidence | required | required | optional | yes | no | no |
| Full token generation | required | comparator | required | required | required | planned |
| Tool and response replay | required | no | required | required | required | planned |
| Structured output | required | no | required | required | required or refused | planned or refused |
| Media | marker/refusal | processor facts | marker/refusal | refused | refused | refused |
| Performance | no | no | required | informational | retained | follow-on retained |

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
5. Qualified GGUF descriptor, memory plan, and refusal tests.
6. Native CPU generation and reference parity.
7. Native CUDA generation and backend-truth receipts.
8. OpenAI-compatible serving, tools, streaming, and response replay.
9. Release checker, comparator bundle, and canonical doc status update.
10. Native Metal generation and Apple backend evidence.

Generated fixtures and reports land with the code that produces them. Do not
mix broad formatting, unrelated Qwen3.5 benchmark files, or incidental report
regeneration into these commits.

## External Inputs And Blockers

R1-R4 can begin with the current repository and official BF16 artifact.

R5 and later need:

- the selected `Qwen3.8-27B-UD-Q3_K_XL.gguf` artifact downloaded and bound to
  immutable provenance, exact byte size, and SHA-256
- enough local GPU margin for the chosen artifact and context window, or a
  larger remote GPU for full-BF16 CUDA validation
- upstream reference outputs for tokenizer, prompt, logits, generation, and
  eventually vision

These inputs do not block the metadata, template, admission, or bounded BF16
milestones. They block honest native generation and accelerated release claims.

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
