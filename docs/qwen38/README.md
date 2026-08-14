# Qwen3.8 Research

> Status: `planned` on 2026-08-14. This directory records upstream facts and
> Psionic implementation analysis. Psionic does not yet claim Qwen3.8
> inference, serving, training, or multimodal support.

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

## Documents

- [MODEL_FACTS.md](MODEL_FACTS.md) records facts read from the upstream model
  card, config, tokenizer, processors, and safetensors index.
- [PSIONIC_GAP_ANALYSIS.md](PSIONIC_GAP_ANALYSIS.md) maps those facts to the
  current Qwen3.5 and Qwen3.6 implementation and defines the first honest
  implementation steps.
- [UPSTREAM_ARTIFACT_INDEX.md](UPSTREAM_ARTIFACT_INDEX.md) links every file in
  the pinned upstream repository and explains its role in loading, serving,
  prompt rendering, preprocessing, or weight materialization.
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) defines the ordered
  implementation milestones, acceptance gates, validation matrix, commit
  sequence, and release claim boundary for Qwen3.8 support.
- [FIRST_GGUF_TARGET.md](FIRST_GGUF_TARGET.md) selects
  `Qwen3.8-27B-UD-Q3_K_XL.gguf` as the first execution artifact and defines its
  local CUDA residency, context, provenance, and comparison gates.

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

## Primary Source

- <https://huggingface.co/Qwen/Qwen3.8-27B>
