# Qwen3.8-27B Upstream Artifact Index

> Status: `planned`. This index describes the 32 files verified at upstream
> revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` on 2026-08-14.

The local artifact root is `target/models/qwen/Qwen3.8-27B`. Links below point
to the pinned Hugging Face revision so they remain stable if upstream `main`
changes.

The hidden local `.cache/huggingface/` directory is Hugging Face CLI download
state. It is not part of the upstream repository and is excluded from this
index.

## Contents

- [Repository metadata](#repository-metadata)
- [Model and generation configuration](#model-and-generation-configuration)
- [Tokenizer and prompt template](#tokenizer-and-prompt-template)
- [Image and video processors](#image-and-video-processors)
- [Weight index and shards](#weight-index-and-shards)
- [Psionic consumption order](#psionic-consumption-order)

## Repository Metadata

| File | Purpose |
| --- | --- |
| [`.gitattributes`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/.gitattributes) | Declares which artifact types use Git LFS. The Qwen repository stores safetensors and `tokenizer.json` as LFS objects. It is repository transport metadata, not a runtime input. |
| [`LICENSE`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) | Apache-2.0 license governing use and redistribution of the released artifact. It must stay associated with any redistributed model bundle. |
| [`README.md`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md) | Official model card. It defines the published architecture summary, capabilities, benchmark claims, quickstart, reasoning controls, context-extension guidance, and operating recommendations. It is documentation, not executable configuration. |
| [`crc32.txt`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/crc32.txt) | CRC32 inventory for the template, generation config, tokenizer assets, and processor configs. It provides a lightweight transport-integrity check for those eight files. Psionic receipts should continue to use cryptographic digests. |

## Model And Generation Configuration

| File | Purpose |
| --- | --- |
| [`config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) | Authoritative model topology and dtype contract. It describes the `Qwen3_5ForConditionalGeneration` wrapper, `qwen3_5_text` decoder, 64-layer hybrid layout, 27-layer vision encoder, dimensions, token ids, RoPE facts, MTP facts, and BF16 storage. Psionic model admission must bind this file and its digest to every execution report. |
| [`generation_config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/generation_config.json) | Published default decoding policy: sampling enabled with temperature 1.0, top-k 20, top-p 0.95, two EOS ids, and the pad id. Request-level settings may override it, but receipts must report the effective values. |

## Tokenizer And Prompt Template

| File | Purpose |
| --- | --- |
| [`chat_template.jinja`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja) | Standalone canonical Jinja chat template. It defines message framing, thinking defaults, `reasoning_effort`, preserved reasoning, tool-call XML, tool responses, image/video markers, and media refusal in system messages. Psionic needs a digest-bound equivalent renderer and fixtures. |
| [`merges.txt`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/merges.txt) | Ordered byte-pair-encoding merge rules used with `vocab.json`. It supports slow or reconstructed tokenizer implementations that do not load the monolithic tokenizer artifact. |
| [`tokenizer.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json) | Complete serialized Hugging Face tokenizer graph. It packages the BPE vocabulary and merges with NFC normalization, pre-tokenization, byte-level post-processing and decoding, plus added tokens. This is the primary tokenizer artifact for direct loading. |
| [`tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json) | Tokenizer class, special-token declarations, 262,144-token maximum length, added-token metadata, and embedded chat template. It supplies template behavior and semantic token identity that the tokenizer graph alone does not fully express. |
| [`vocab.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/vocab.json) | BPE token-to-id vocabulary used with `merges.txt`. It is the decomposed vocabulary source for tokenizer implementations that do not consume `tokenizer.json` directly. |

## Image And Video Processors

| File | Purpose |
| --- | --- |
| [`preprocessor_config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/preprocessor_config.json) | Image preprocessing contract for `Qwen3VLProcessor` and `Qwen2VLImageProcessorFast`: image bounds, normalization, patch size, temporal patch size, and merge size. Prompt marker projection does not consume this file; native vision execution must. |
| [`video_preprocessor_config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/video_preprocessor_config.json) | Video preprocessing contract for `Qwen3VLVideoProcessor`: frame-input bounds, normalization, spatial patches, temporal patches, and merge size. Native video understanding requires this processor plus an explicit frame-sampling policy. |

## Weight Index And Shards

[`model.safetensors.index.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json)
is the authoritative tensor-to-shard map. It assigns 1,199 tensor names to 18
files and reports 55,562,855,904 tensor-data bytes. Loaders must use this index
instead of assuming layer-contiguous or numerically ordered shard contents.

Shard boundaries are packaging choices. Several language layers span two
shards, shard 1 mixes all vision tensors with early language layers, shard 3
contains only the token embedding, and shard 18 contains the LM head and MTP
tensors.

| File | Size | Tensors | Indexed contents |
| --- | ---: | ---: | --- |
| [`model-00001-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00001-of-00018.safetensors) | 3,966,730,552 | 392 | All 333 vision tensors plus portions of language layers 0-4. |
| [`model-00002-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00002-of-00018.safetensors) | 3,043,080,328 | 47 | Remaining or partial tensors for language layers 4-7. |
| [`model-00003-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00003-of-00018.safetensors) | 2,542,796,952 | 1 | `model.language_model.embed_tokens.weight` only. |
| [`model-00004-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00004-of-00018.safetensors) | 3,988,973,152 | 69 | Portions of language layers 10-15. |
| [`model-00005-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00005-of-00018.safetensors) | 2,099,339,864 | 37 | Language layers 8-9 and additional portions of layers 10-15. |
| [`model-00006-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00006-of-00018.safetensors) | 3,979,553,696 | 76 | Portions of language layers 16-21. |
| [`model-00007-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00007-of-00018.safetensors) | 2,108,759,344 | 30 | Remaining or partial tensors for language layers 21-23. |
| [`model-00008-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00008-of-00018.safetensors) | 3,979,553,696 | 76 | Portions of language layers 24-29. |
| [`model-00009-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00009-of-00018.safetensors) | 2,108,759,344 | 30 | Remaining or partial tensors for language layers 29-31. |
| [`model-00010-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00010-of-00018.safetensors) | 3,979,553,696 | 76 | Portions of language layers 32-37. |
| [`model-00011-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00011-of-00018.safetensors) | 2,108,759,344 | 30 | Remaining or partial tensors for language layers 37-39. |
| [`model-00012-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00012-of-00018.safetensors) | 3,979,553,696 | 76 | Portions of language layers 40-45. |
| [`model-00013-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00013-of-00018.safetensors) | 2,108,759,344 | 30 | Remaining or partial tensors for language layers 45-47. |
| [`model-00014-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00014-of-00018.safetensors) | 3,979,553,696 | 76 | Portions of language layers 48-53. |
| [`model-00015-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00015-of-00018.safetensors) | 2,108,759,344 | 30 | Remaining or partial tensors for language layers 53-55. |
| [`model-00016-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00016-of-00018.safetensors) | 3,979,564,040 | 77 | Portions of language layers 56-61 plus `model.language_model.norm.weight`. |
| [`model-00017-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00017-of-00018.safetensors) | 2,108,759,344 | 30 | Remaining or partial tensors for language layers 61-63. |
| [`model-00018-of-00018.safetensors`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model-00018-of-00018.safetensors) | 3,392,197,344 | 16 | `lm_head.weight` plus all 15 tensors under the single MTP layer. |

## Psionic Consumption Order

An honest loader should consume these files in this order:

1. Bind the repository revision, license, and artifact digests.
2. Parse `config.json` and admit a Qwen3.8-specific product identity over the
   reusable `qwen3_5_text` architecture contract.
3. Load `tokenizer.json` with `tokenizer_config.json`, then verify the separate
   `chat_template.jinja` digest and behavior fixtures.
4. Parse `model.safetensors.index.json`, validate its complete tensor map, and
   open shards by the filenames in that map rather than numeric assumptions.
5. Load only the text tensors for a declared text-only lane. Report all ignored
   vision tensors and refuse image/video inputs on that lane.
6. Load both processor configs and vision tensors only for an explicitly
   admitted native multimodal lane.
7. Apply `generation_config.json` as published defaults while retaining
   request-level effective sampling truth in execution receipts.
