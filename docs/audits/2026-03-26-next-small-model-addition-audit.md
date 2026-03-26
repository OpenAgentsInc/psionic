# 2026-03-26 Next Small Model Addition Audit

## Intent

This audit now answers the active question:

> if Psionic should support `qwen35` GGUFs starting with the small Ollama
> `qwen3.5:0.8b` row, what exact Psionic changes are required?

This supersedes the earlier "least new substrate work" framing.

The active requirement is explicit `Qwen3.5` support.

## Decision

Psionic should start `qwen35` support with the Ollama `qwen3.5:0.8b` GGUF.

That row is the correct first target because:

- it matches the laptop requirement from `smallmodels.md`
- it is current Qwen3.5, not an older `qwen2` or `qwen2.5` row
- it is already available as a public GGUF artifact through the Ollama registry
- it forces Psionic to add the real `qwen35` substrate instead of hiding behind
  old `qwen2` assumptions

The exact downloaded artifact used for this audit is:

- manifest: `https://registry.ollama.ai/v2/library/qwen3.5/manifests/0.8b`
- manifest digest: `f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4`
- model blob:
  `https://registry.ollama.ai/v2/library/qwen3.5/blobs/sha256:afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5`
- local path:
  `/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf`
- local SHA-256:
  `afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5`
- on-disk size: about `989M`

## Sources Reviewed

Local planning input:

- `~/code/alpha/psionic/smallmodels.md`

Canonical Psionic docs:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/INFERENCE_ENGINE.md`
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
- `docs/MLX_MODEL_CATALOG.md`
- `docs/MLX_TEXT_SERVE.md`
- `docs/NON_GPT_OSS_QWEN_PILOT.md`

Relevant Psionic code paths:

- `crates/psionic-models/src/lib.rs`
- `crates/psionic-models/src/runtime_tokenizer.rs`
- `crates/psionic-models/src/fixtures.rs`
- `crates/psionic-serve/src/gguf.rs`
- `crates/psionic-serve/src/openai_http.rs`
- `crates/psionic-serve/src/conformance.rs`

Relevant Ollama code paths:

- `~/code/ollama/x/create/create.go`
- `~/code/ollama/server/internal/client/ollama/registry.go`
- `~/code/ollama/x/models/qwen3_5/qwen3_5.go`

External model references:

- `https://ollama.com/library/qwen3.5:0.8b`
- `https://registry.ollama.ai/v2/library/qwen3.5/manifests/0.8b`
- `https://huggingface.co/Qwen/Qwen3.5-0.8B`
- `https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/chat_template.jinja`
- `https://huggingface.co/docs/transformers/model_doc/qwen3_5`

## Why `qwen3.5:0.8b` Is The Right First `qwen35` Row

`smallmodels.md` names one tester machine directly:

- `8GB` NVIDIA laptop VRAM
- `32GB` system RAM

`qwen3.5:0.8b` fits that target.

The Ollama blob is about `0.96 GiB` in `Q8_0`. That is small enough to serve as
the first real `qwen35` bring-up row without turning the effort into a memory
project before it becomes a model-support project.

This is also the right target because it is not just a tiny legacy chat model.

It is a real `Qwen3.5` row with the current hybrid architecture, current chat
template, current tokenizer pretokenizer, current long-context metadata, and
current multimodal metadata. If Psionic can load and serve this row honestly,
Psionic has real `qwen35` support instead of a compatibility label built on old
`qwen2` assumptions.

## What The Downloaded GGUF Actually Contains

The downloaded GGUF is not a renamed `qwen2` artifact.

The file carries these direct facts:

- `general.architecture = "qwen35"`
- `tokenizer.ggml.pre = "qwen35"`
- `qwen35.block_count = 24`
- `qwen35.context_length = 262144`
- `qwen35.embedding_length = 1024`
- `qwen35.feed_forward_length = 3584`
- `qwen35.full_attention_interval = 4`
- `qwen35.ssm.conv_kernel = 4`
- `qwen35.ssm.group_count = 16`
- `qwen35.ssm.inner_size = 2048`
- `qwen35.ssm.state_size = 128`
- `qwen35.ssm.time_step_rank = 16`
- `qwen35.vision.block_count = 12`
- `qwen35.vision.embedding_length = 768`
- `qwen35.vision_start_token_id = 248053`
- `qwen35.vision_end_token_id = 248054`
- default chat-template digest:
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`

The tensor inventory also shows that the language stack is hybrid.

The first observed block pattern is:

- `blk.0`, `blk.1`, `blk.2`:
  `attn_qkv.weight`, `attn_gate.weight`, `ffn_*`, and `ssm_*`
- `blk.3`:
  `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`,
  and `ffn_*`
- then the same `3 hybrid + 1 full-attention` pattern repeats

That matches `qwen35.full_attention_interval = 4`.

This matters because Psionic does not currently have a decoder-family contract
for a mixed SSM plus attention stack.

## Where Psionic Fails Today

Psionic does not need one change.

Psionic needs a stack of coordinated changes.

### 1. The loader rejects `qwen35` immediately

`crates/psionic-models/src/lib.rs` only maps these decoder architectures today:

- `llama`
- `mistral`
- `mistral3`
- `qwen2`
- `gpt-oss`

`qwen35` is rejected before Psionic even reaches prompt or tensor validation.

### 2. The tokenizer layer only knows `qwen2`

`GgufTokenizerPretokenizer` only has `Qwen2`.

`runtime_tokenizer.rs` only accepts:

- `qwen2`
- `default`
- `llama-bpe`
- `llama`
- a few other existing strings

So even a metadata-only admission patch is incomplete without a `qwen35`
pretokenizer path.

### 3. Prompt rendering only knows the old `qwen2` digest

Psionic only recognizes the existing `qwen2` prompt-template digest in
`supported_prompt_template_family`.

The real `qwen35` chat template is different:

- it carries multimodal placeholders
- it carries tool-call XML formatting branches
- it carries explicit `<think>` handling
- it uses the `qwen35` tokenizer family

The actual default template digest for the downloaded GGUF is:

- `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`

Without a new prompt family, Psionic cannot honestly render a `qwen35` request
through the generic server.

### 4. The current decoder tensor layout cannot represent `qwen35`

This is the deeper blocker.

`GgufDecoderLayerTensorLayout` currently requires every decoder layer to carry:

- `attention_query_weight`
- `attention_key_weight`
- `attention_value_weight`
- `attention_output_weight`

Those are required `String` fields, not optional fields.

There is also no place in that struct for:

- `attn_gate.weight`
- `attn_qkv.weight`
- `ssm_a`
- `ssm_dt`
- `ssm_alpha.weight`
- `ssm_beta.weight`
- `ssm_conv1d.weight`
- `ssm_norm.weight`
- `ssm_out.weight`

That means a real `qwen35` hybrid block does not fit the current Psionic layer
layout contract.

### 5. The current CPU dense runtime is structurally wrong for `qwen35`

`crates/psionic-serve/src/gguf.rs` implements one dense non-GPT-OSS path:

- RMS norm
- attention with `Q/K/V`
- output projection
- SiLU-GLU feed-forward

That is a regular transformer decoder lane.

It has no execution path for:

- fused gated QKV on hybrid blocks
- SSM state update
- SSM convolution
- alternating `3 hybrid + 1 full-attention` block scheduling

If Psionic simply adds `qwen35` to the loader classifier today, the next hard
failure is not a small shape tweak.

The next hard failure is structural. The current builder would try to load
`blk.0.attn_q.weight` from a block that actually exposes `blk.0.attn_qkv.weight`
plus `ssm_*`.

That is an inference from the source and the downloaded tensor inventory.

### 6. The artifact is multimodal, and Psionic must not hide that fact

The downloaded GGUF carries:

- `qwen35.vision.*` metadata
- vision start/end token ids
- a multimodal chat template

Psionic can still choose a text-only first slice.

Psionic must not pretend that a multimodal artifact is already fully admitted.

The first row should be:

- language-model-only execution
- explicit refusal for image and video inputs
- explicit refusal for tool-call formatting if Psionic has not implemented it

That refusal posture needs to be part of the design, not an undocumented
omission.

## Exact Psionic Changes Needed

The clean implementation is a seven-part change.

### 1. Add a real `qwen35` family admission path in `psionic-models`

Files:

- `crates/psionic-models/src/lib.rs`
- `crates/psionic-models/src/runtime_tokenizer.rs`
- `crates/psionic-models/src/sharding.rs`

Required changes:

- Add `qwen35` to GGUF architecture admission.
- Add a distinct `GgufTokenizerPretokenizer::Qwen35`.
- Accept `qwen35` in the runtime tokenizer byte-level BPE path.
- Add a distinct family label instead of silently folding this into old
  `qwen2` metadata.

The correct shape is:

- add `GgufDecoderFamily::Qwen35`
- keep `GgufDecoderFamily::Qwen` for current `qwen2`

That split matters because:

- the architecture string is different
- the prompt family is different
- the block layout is different
- the multimodal metadata is different

Trying to overload `Qwen` here would leak ambiguity across serving, runtime
support, and evidence surfaces.

### 2. Add a bounded `qwen35` prompt family

Files:

- `crates/psionic-models/src/lib.rs`
- `crates/psionic-models/src/fixtures.rs`
- `crates/psionic-serve/src/openai_http.rs`

Required changes:

- Add `GgufPromptTemplateFamily::Qwen35`.
- Recognize the downloaded template digest
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`.
- Implement a bounded `render_qwen35_text_only` path for:
  - optional leading `system`
  - `user`
  - `assistant`
  - `add_generation_prompt`
- Match the current Qwen3.5 text-only template behavior:
  - `system` stays first
  - text turns render with `<|im_start|>` / `<|im_end|>`
  - generation prompt opens the assistant turn
  - non-thinking mode emits the empty `<think>` scaffold before completion

The first slice should explicitly refuse:

- image content
- video content
- `tool` messages
- `tool_calls`
- assistant reasoning-content replay if Psionic has not implemented it yet

That is enough to make `/v1/chat/completions` honest for text-only requests
without promising the whole Qwen3.5 prompt surface on day one.

### 3. Replace the current uniform decoder-layer layout with a hybrid block layout

Files:

- `crates/psionic-models/src/lib.rs`

Required changes:

- Extend `GgufDecoderFamilyMetadata` with `qwen35`-specific execution metadata:
  - `full_attention_interval`
  - `ssm_conv_kernel`
  - `ssm_group_count`
  - `ssm_inner_size`
  - `ssm_state_size`
  - `ssm_time_step_rank`
  - optional vision metadata fields preserved for refusal/reporting
- Replace or extend `GgufDecoderLayerTensorLayout` so one layer can be either:
  - full-attention block
  - hybrid SSM block

At minimum the hybrid layout needs slots for:

- `attn_gate.weight`
- `attn_qkv.weight`
- `attn_qkv.bias` if present
- `post_attention_norm.weight`
- `ssm_a`
- `ssm_dt`
- `ssm_alpha.weight`
- `ssm_beta.weight`
- `ssm_conv1d.weight`
- `ssm_norm.weight`
- `ssm_out.weight`

The current decoder layout cannot hold those tensors honestly.

Do not patch around this by inventing fake `attn_q.weight` names.

### 4. Add a `qwen35` execution config instead of forcing it into the old dense decoder contract

Files:

- `crates/psionic-models/src/lib.rs`
- `crates/psionic-serve/src/gguf.rs`
- any shared decoder-config crate types reached through
  `DecoderModelDescriptor`

Required changes:

- Add a qwen35-specific execution config or extend the generic decoder config
  with per-layer block kinds.
- Record the alternating block schedule instead of pretending all layers are the
  same transformer block.

The real artifact is:

- `24` total layers
- `full_attention_interval = 4`
- effectively `6` groups of `3 hybrid blocks + 1 full-attention block`

Psionic should preserve that truth directly in machine-readable metadata.

### 5. Add a dedicated `qwen35` CPU runtime instead of routing it through the current dense path

Files:

- `crates/psionic-serve/src/gguf.rs`

Required changes:

- Add a new runtime kind such as `CpuQwen35GgufTextGenerationService`.
- Route `GgufDecoderFamily::Qwen35` into that runtime, not into
  `CpuDenseGgufTextGenerationService`.
- Implement:
  - hybrid-block execution
  - SSM state update
  - SSM convolution
  - fused gated-QKV handling on hybrid blocks
  - full-attention handling on the `interval = 4` blocks
  - tied-embedding LM head behavior

This should start as:

- CPU only
- text only
- single-family bring-up

Do not claim:

- CUDA support
- Metal support
- adapter hosting
- throughput parity
- multimodal execution

until those surfaces are actually validated.

### 6. Add explicit server refusals for the unsupported `qwen35` surfaces

Files:

- `crates/psionic-serve/src/openai_http.rs`
- `crates/psionic-serve/src/conformance.rs`

Required changes:

- Reject image/video request parts for `qwen35` until a real multimodal encoder
  path lands.
- Reject tool-call surfaces until the `qwen35` tool XML format is implemented.
- Keep model inventory honest:
  - `psionic_model_family = "qwen35"` is better than collapsing this into
    `"qwen"`
  - also surface `general.architecture = "qwen35"`
  - include the chat-template digest
  - include the refusal reasons for multimodal and tools in machine-readable
    form when relevant

That keeps the server truthful while still making the first text path usable.

### 7. Add real qwen35 fixtures, conformance, and release checks

Files:

- `crates/psionic-models/src/fixtures.rs`
- `crates/psionic-models/src/lib.rs` tests
- `crates/psionic-serve/src/conformance.rs`
- `crates/psionic-serve/src/gguf.rs` tests
- `crates/psionic-serve/src/openai_http.rs` tests
- `scripts/release/check-psionic-qwen35-pilot.sh`
- `docs/NON_GPT_OSS_QWEN35_PILOT.md`

Required changes:

- Add a real `qwen35` tokenizer fixture.
- Add a real `qwen35` prompt-template digest fixture.
- Add a synthetic tiny `qwen35` GGUF metadata fixture that includes:
  - `general.architecture = qwen35`
  - `tokenizer.ggml.pre = qwen35`
  - the real `qwen35` chat-template digest
  - one hybrid block and one full-attention block pattern
- Add loader tests for:
  - architecture admission
  - tokenizer admission
  - prompt rendering
  - hybrid tensor-layout construction
- Add runtime tests for:
  - hybrid block execution on a deterministic tiny fixture
  - generic server inventory and refusal signals
- Add a release checker for the real downloaded row:
  - local GGUF path or Ollama manifest input
  - machine-checkable text-only request
  - refusal checks for unsupported multimodal inputs

The first pilot should prove correctness and honesty.

It should not pretend to prove broad Qwen3.5 completion.

## Recommended Implementation Order

The clean order is:

1. Add `qwen35` architecture admission, tokenizer admission, and prompt-family
   admission.
2. Add a hybrid decoder metadata/layout contract in `psionic-models`.
3. Add a dedicated CPU `qwen35` runtime in `psionic-serve`.
4. Add text-only generic-server support with explicit refusals for multimodal
   and tools.
5. Add the `qwen35` pilot checker and docs.
6. Only after that, consider:
   - tool-call formatting
   - multimodal input support
   - adapters
   - GPU rows

## What Psionic Should Not Do

Psionic should not:

- map `qwen35` to `qwen2` and call that support
- advertise a multimodal row while silently serving text only
- reuse the old Qwen prompt family digest
- stuff hybrid SSM blocks into the old dense decoder layout with fake tensor
  names
- claim throughput or backend portability before there is a real pilot row

## Final Recommendation

The right next model row is:

- **`qwen3.5:0.8b` as the first explicit `qwen35` GGUF target**

The exact Psionic work required is also clear:

- add a real `qwen35` family and tokenizer path
- add a bounded `qwen35` prompt renderer for text-only chat
- add a hybrid SSM plus attention decoder layout contract
- add a dedicated `qwen35` CPU runtime
- add explicit refusal posture for multimodal and tool surfaces
- add fixtures, conformance, and a release checker for the real row

If Psionic wants one sentence to guide the implementation tranche, it should be
this:

> Add `qwen3.5:0.8b` next, and do the real `qwen35` work: new family
> admission, new prompt digest, hybrid SSM plus attention layer contracts, a
> dedicated text-only runtime, and explicit refusal on the multimodal surfaces
> that are not implemented yet.
