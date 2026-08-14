# Qwen3.8 Research

> Status: `planned` on 2026-08-14. Upstream research, local artifact
> acquisition, R1 product/artifact identity, R2 prompt/tokenizer contracts, and
> R3 checkpoint admission are `implemented`. Psionic does not yet claim
> Qwen3.8 inference, serving, training, or multimodal support.

This directory tracks the work required to add honest Qwen3.8 support to
Psionic.

## Official Reference Target

The official dense post-trained checkpoint is the source-of-truth target:

- model: `Qwen/Qwen3.8-27B`
- upstream repository revision observed on 2026-08-14:
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- upstream license: Apache-2.0
- pipeline: image-text-to-text
- native context length: 262,144 tokens

## First Execution Target

The first qualified execution artifact will be
`Qwen3.8-27B-UD-Q3_K_XL.gguf` from
`unsloth/Qwen3.8-27B-GGUF` at the observed revision
`fdd03b8bbd279c1694563650e79d85a2373d9934`.

This 13.4 GB mixed-quantization artifact is the primary candidate for a
fully-resident text lane on the local 16 GiB RTX 4080. The first CUDA gate is
4,096 tokens, followed by a separately measured 8,192-token gate. The
selection and its acceptance requirements are defined in
[FIRST_GGUF_TARGET.md](FIRST_GGUF_TARGET.md).

## Local Artifact

The complete upstream repository is available locally at:

```text
target/models/qwen/Qwen3.8-27B
```

The download is pinned to the revision above. Hugging Face CLI verification
passed for all 32 repository files, including all 18 weight shards. The
directory is ignored by Git through `/target/`; model weights are not part of
the repository.

The selected Unsloth GGUF and its three small repository companions are
materialized separately at:

```text
target/models/qwen/unsloth/Qwen3.8-27B-GGUF
```

The selected file is exactly 13,441,059,904 bytes with SHA-256
`00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2`.
Other quantizations, BF16 GGUF shards, and vision projectors are not downloaded.
Their immutable sizes, digests, purposes, and materialization posture are
recorded in
[UNSLOTH_GGUF_ARTIFACT_INDEX.md](UNSLOTH_GGUF_ARTIFACT_INDEX.md).

## Documents

- [MODEL_FACTS.md](MODEL_FACTS.md) records facts read from the upstream model
  card, config, tokenizer, processors, and safetensors index.
- [PSIONIC_GAP_ANALYSIS.md](PSIONIC_GAP_ANALYSIS.md) maps those facts to the
  current Qwen3.5 and Qwen3.6 implementation and defines the first honest
  implementation steps.
- [UPSTREAM_ARTIFACT_INDEX.md](UPSTREAM_ARTIFACT_INDEX.md) links every file in
  the pinned upstream repository and explains its role in loading, serving,
  prompt rendering, preprocessing, or weight materialization.
- [UNSLOTH_GGUF_ARTIFACT_INDEX.md](UNSLOTH_GGUF_ARTIFACT_INDEX.md) indexes the
  complete pinned Unsloth GGUF tree, records exact sizes and LFS digests, and
  distinguishes the selected local text artifact from deferred comparators,
  BF16 shards, and vision projectors.
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) defines the ordered
  implementation milestones, acceptance gates, validation matrix, commit
  sequence, and release claim boundary for Qwen3.8 support.
- [FIRST_GGUF_TARGET.md](FIRST_GGUF_TARGET.md) selects
  `Qwen3.8-27B-UD-Q3_K_XL.gguf` as the first execution artifact and defines its
  local CUDA residency, context, provenance, and comparison gates.
- [UNSLOTH_CODE_AUDIT.md](UNSLOTH_CODE_AUDIT.md) audits the pinned Unsloth
  loader and Studio GGUF paths, identifies the reusable artifact, admission,
  memory, and prompt-handling patterns, and records the license and native
  execution boundaries.
- [LLAMA_CPP_CODE_AUDIT.md](LLAMA_CPP_CODE_AUDIT.md) audits the pinned
  llama.cpp converter, `qwen35` graph, hybrid memory, tokenizer, quantization,
  CPU/CUDA operations, MTP, vision split, and tests, then maps the concrete
  findings to Psionic gates.

## Issue Backlog

[GitHub issue #1157](https://github.com/OpenAgentsInc/psionic/issues/1157)
tracks the complete implementation. Phase issues
[#1143](https://github.com/OpenAgentsInc/psionic/issues/1143) through
[#1156](https://github.com/OpenAgentsInc/psionic/issues/1156) define R1-R13,
including the separate R9A MTP gate. Each issue closes only after its code and
evidence are merged and pushed to `main`.

## Claim Boundary

The upstream checkpoint exists and its metadata is structurally close to
Qwen3.6-27B. The local Qwen3.6 code is still bound to Qwen3.6 model ids,
template ids, schemas, reports, and claim text. Metadata similarity does not
admit Qwen3.8 automatically.

The first implementation must keep all of the following identities explicit:

- upstream repository revision
- model id and served model id
- config, tokenizer, template, processor, index, and shard digests
- execution backend and bounded execution mode
- text-only versus native vision execution
- native 262,144-token context versus any explicitly configured YaRN extension

R1 enforces that boundary through
`fixtures/qwen38/qwen38_27b_artifact_facts_v1.json` and the exported
`psionic_models::Qwen38ArtifactFacts` and
`psionic_models::Qwen38ArtifactAdmissionResult` types. The official and short
27B model ids normalize deterministically. Served ids, other sizes, other
products, and drifted fixtures refuse rather than inheriting the 27B contract.

R2 adds the digest-bound `qwen3.8.chat_template.v1` renderer, prompt receipts,
official-tokenizer loader, retained upstream golden hashes and token ids, and a
dedicated GGUF `qwen35` pretokenizer. It implements NFC normalization,
per-code-point Unicode numeric splitting, reasoning effort, thinking
preservation, tools, grouped results, and media-marker framing. Media markers
remain prompt bytes only; they do not widen the execution claim.

The same fixture retains a pinned `llama-tokenize` comparison. Eight of nine
cases match the official tokenizer exactly. llama.cpp does not NFC-normalize
the decomposed-accent case when reading the selected GGUF, so Psionic keeps the
official normalized IDs and records llama.cpp's divergent IDs explicitly.

R3 extracts the reusable `qwen3_5_text` architecture and safetensors-header
contract into `psionic_models::Qwen35TextArchitectureReport` and related
family-neutral types. Qwen3.6 retains its public wrapper names. The distinct
`psionic_models::Qwen38ForwardAdmissionReport` binds the R1 config and index
digests, derives 18 shards from the index, admits 851 decoder plus 15 MTP
tensors, inventories 333 non-text tensors, and records eight split-layer shard
resolutions. This is checkpoint structure admission, not execution.

## Primary Source

- <https://huggingface.co/Qwen/Qwen3.8-27B>
