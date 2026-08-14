# Qwen3.8 Psionic Gap Analysis

> Status: `planned` on 2026-08-14. This document defines an implementation
> direction. It does not claim that Qwen3.8 executes in Psionic.

## Compatibility Result

Qwen3.8-27B and Qwen3.6-27B publish the same text and vision architecture
dimensions. Their normalized `config.json` files differ only in the declared
Transformers version:

- Qwen3.6-27B: `4.57.1`
- Qwen3.8-27B: `5.8.0.dev0`

The two repositories also have:

- the same `qwen3_5` root model type
- the same `qwen3_5_text` text model type
- the same 1,199 tensor names
- the same 55,562,855,904 indexed tensor bytes
- the same generation config
- byte-identical image and video processor configs

The packaging differs: Qwen3.6-27B uses 15 safetensors shards and Qwen3.8-27B
uses 18. Weight-map tensor names are identical, while the shard assignments and
index digest differ.

This supports reuse of a family-level dense `qwen3_5_text` architecture
contract. It does not support treating Qwen3.8 as a Qwen3.6 alias.

## Existing Reusable Psionic Work

### Qwen3.6 safetensors path

`crates/psionic-models/src/qwen36_forward_admission.rs` already understands the
published dense 27B tensor layout. It provides:

- config and tensor-table admission
- shape and dtype validation
- shard-header inspection
- sampled embedding/LM-head projection
- bounded row-sparse traversal through every declared text layer
- MTP tensor lookup
- deterministic receipts and typed refusal

The current implementation is hard-coded to Qwen3.6 model ids, schema names,
errors, report text, and local paths. Qwen3.8 must not enter through those
identities unchanged.

### Qwen3.5 native runtime

`crates/psionic-serve/src/qwen35.rs` already contains CPU, CUDA, and Metal GGUF
execution for the Qwen3.5 hybrid layout. It includes linear-attention and
full-attention blocks, quantized projections, token generation, streaming,
sampling, structured-output support, and cache machinery.

That path consumes GGUF family facts and does not directly load the official
Qwen3.8 BF16 safetensors checkpoint. Kernel reuse requires an explicit bridge
between admitted Hugging Face tensor semantics and the runtime's model plan.

### Prompt and server surfaces

The generic OpenAI-compatible server already has Qwen-family tool calling,
streaming, response-state replay, and prompt-level image/video projection.
Qwen3.8 changes default reasoning and preservation behavior, so existing
Qwen3.5 or Qwen3.6 template behavior cannot be reused without a Qwen3.8
fixture and explicit request controls.

## Required Identity Model

The implementation should separate architecture identity from product model
identity:

- architecture family: `qwen3_5_text`
- product family: `qwen38`
- artifact model id: `Qwen/Qwen3.8-27B`
- served model id: `qwen3.8-27b`
- template id: a new Qwen3.8-specific version

This permits kernel and tensor-contract reuse without emitting Qwen3.6 schema
names or receipts for a Qwen3.8 artifact.

## Initial Implementation Sequence

### 1. Artifact fact fixture

Add a small committed fixture containing the official config, tokenizer facts,
template digest, processor facts, and safetensors-index inventory. Do not
commit model weights.

Acceptance:

- exact upstream revision and file digests are recorded
- model identity normalizes only the official and explicit short ids
- architecture, tensor count, tensor-byte count, and shard count are checked
- tokenizer and template identities remain distinct from Qwen3.6

### 2. Family-neutral dense checkpoint admission

Extract the reusable `qwen3_5_text` tensor specification from the Qwen3.6-only
admission code, then add Qwen3.8-specific report and schema wrappers.

Acceptance:

- the official Qwen3.8 index admits all expected text tensors
- vision and other non-text tensors remain separately inventoried
- missing tensors, shape drift, dtype drift, and unexpected model ids refuse
- reports say Qwen3.8 and never claim Qwen3.6 execution

### 3. Qwen3.8 prompt contract

Implement and fixture the published template semantics:

- thinking enabled by default
- `reasoning_effort` validation and instruction injection
- `preserve_thinking` enabled by default
- explicit non-thinking mode
- assistant reasoning preservation
- tool calls and grouped tool results
- image/video marker rendering and system-message media refusal

The fixture should render cases from parsed template semantics rather than
copying the upstream Jinja file into a handwritten approximation without
tests.

### 4. Bounded real-checkpoint execution

Run the existing header-admission, sampled-projection, and row-sparse
full-layer evidence levels against the official 18-shard checkpoint. Keep the
same limitations currently attached to the Qwen3.6 row-sparse path.

Acceptance:

- real BF16 rows are read from the Qwen3.8 shards
- every declared text layer is visited in order
- tensor-read and sampled-logit digests are deterministic
- execution reports do not claim full-width inference or generation

### 5. Production text generation

Connect the admitted Qwen3.8 tensor layout to native CPU, CUDA, and Metal
execution only after parity fixtures prove the kernel contract. Validate the
full-attention, Gated DeltaNet, FFN, normalization, RoPE, cache, LM-head, and
MTP boundaries independently.

Acceptance requires real multi-token generation, deterministic replay under a
fixed sampling plan, backend truth, refusal without fallback, and comparator
evidence against an upstream-supported runtime.

### 6. Native vision

Treat native vision as a separate capability. Prompt projection of image and
video markers must not be published as image or video understanding.

Acceptance requires the 27-layer vision encoder, image/video preprocessing,
vision-to-text projection, attachment receipts, and end-to-end multimodal
parity evidence.

## Known Risks

- Upstream used a development Transformers version at publication time.
  Psionic must rely on artifact facts, not an unstable Python dependency.
- The Qwen3.8 tokenizer differs from Qwen3.6 despite matching special-token
  ids and vocabulary size.
- Default preserved reasoning changes prompt length and cache identity across
  multi-turn conversations.
- A 262,144-token native window creates large cache and memory requirements.
  Any 1,000,000-token YaRN lane needs separate admission and memory truth.
- The official checkpoint is multimodal. Text-only admission must report that
  it is ignoring or refusing the vision tensors instead of labeling the whole
  model text-only.
- Exact architecture equality in config does not prove numerical equivalence
  of an existing Qwen3.5 GGUF kernel path against the new BF16 weights.

## Current Refusal Posture

Until the first artifact fixture and model-id admission land, Psionic should
refuse `Qwen/Qwen3.8-27B` as unsupported. It should not silently normalize the
model to Qwen3.6, Qwen3.5, or generic Qwen.
