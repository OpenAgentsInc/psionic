# Qwen3.8 Research

> Status: `planned` on 2026-08-14. This directory records upstream facts and
> Psionic implementation analysis. Psionic does not yet claim Qwen3.8
> inference, serving, training, or multimodal support.

This directory tracks the work required to add honest Qwen3.8 support to
Psionic.

## Current Target

The first target is the official dense post-trained checkpoint:

- model: `Qwen/Qwen3.8-27B`
- upstream repository revision observed on 2026-08-14:
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- upstream license: Apache-2.0
- pipeline: image-text-to-text
- native context length: 262,144 tokens

## Documents

- [MODEL_FACTS.md](MODEL_FACTS.md) records facts read from the upstream model
  card, config, tokenizer, processors, and safetensors index.
- [PSIONIC_GAP_ANALYSIS.md](PSIONIC_GAP_ANALYSIS.md) maps those facts to the
  current Qwen3.5 and Qwen3.6 implementation and defines the first honest
  implementation steps.

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
