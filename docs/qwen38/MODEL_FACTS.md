# Qwen3.8-27B Model Facts

> Status: checkpoint facts and header admission are `implemented`; execution is
> `planned`. Facts were read from the official Hugging Face repository at
> revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` on 2026-08-14.

## Artifact Identity

| Field | Value |
| --- | --- |
| Model id | `Qwen/Qwen3.8-27B` |
| Repository visibility | public, ungated |
| License | Apache-2.0 |
| Pipeline tag | `image-text-to-text` |
| Library | Hugging Face Transformers |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Root model type | `qwen3_5` |
| Text model type | `qwen3_5_text` |
| Weight format | BF16 safetensors |
| Weight shards | 18 |
| Indexed tensors | 1,199 |
| Indexed tensor bytes | 55,562,855,904 |

The model card calls this a 27B causal language model with a vision encoder.
The checkpoint is post-trained and includes both text and native image/video
components.

The R3 family-neutral tensor specification and the complete official header
pass classify the 1,199 indexed tensors as follows:

| Inventory | Count |
| --- | ---: |
| Decoder trunk, embeddings, norm, and LM head | 851 |
| MTP projection, norm, and appended layer | 15 |
| Vision or other non-text tensors | 333 |

The index resolves 18 shards. Decoder layers `4`, `15`, `21`, `29`, `37`,
`45`, `53`, and `61` span two shard files; their shard sets are retained in the
Qwen3.8 forward-admission report. Header admission validates names, dtypes,
shapes, and exact tensor-to-shard mapping without reading tensor payloads.

## Text Architecture

| Field | Value |
| --- | ---: |
| Hidden width | 5,120 |
| Layers | 64 |
| Vocabulary size | 248,320 |
| FFN intermediate width | 17,408 |
| Native maximum positions | 262,144 |
| Full-attention interval | 4 |
| Full-attention query heads | 24 |
| Full-attention KV heads | 4 |
| Full-attention head width | 256 |
| Partial rotary factor | 0.25 |
| RoPE theta | 10,000,000 |
| Linear-attention QK heads | 16 |
| Linear-attention value heads | 48 |
| Linear-attention head width | 128 |
| Linear convolution width | 4 |
| MTP hidden layers in config | 1 |

The 64-layer layout repeats three linear-attention layers followed by one
full-attention layer 16 times. The model card names the linear-attention block
Gated DeltaNet and the full-attention block Gated Attention. Every block is
followed by an FFN.

The text config also declares:

- `attn_output_gate = true`
- `output_gate_type = "swish"`
- `mamba_ssm_dtype = "float32"`
- interleaved MRoPE sections `[11, 11, 10]`
- untied token embeddings and LM output weights
- cache support

## Vision Architecture

| Field | Value |
| --- | ---: |
| Vision depth | 27 |
| Vision hidden width | 1,152 |
| Vision output width | 5,120 |
| Vision attention heads | 16 |
| Vision FFN intermediate width | 4,304 |
| Spatial patch size | 16 |
| Temporal patch size | 2 |
| Spatial merge size | 2 |
| Vision positions | 2,304 |

The image processor identifies itself as `Qwen2VLImageProcessorFast`, and the
video processor identifies itself as `Qwen3VLVideoProcessor`. Both are exposed
through `Qwen3VLProcessor`.

The released image and video processor files are byte-identical to the current
Qwen3.6-27B processor files. This is a compatibility signal, not proof that the
model weights or resulting vision activations are interchangeable.

## Tokenizer And Template

The tokenizer class is `Qwen2Tokenizer`, with a declared maximum length of
262,144. The relevant ids are:

| Token | Id |
| --- | ---: |
| `<|endoftext|>` | 248044 |
| `<|im_start|>` | 248045 |
| `<|im_end|>` | 248046 |
| `<|vision_start|>` | 248053 |
| `<|vision_end|>` | 248054 |
| `<|image_pad|>` | 248056 |
| `<|video_pad|>` | 248057 |
| `<tool_call>` | 248058 |
| `<tool_response>` | 248066 |
| `<think>` | 248068 |
| `</think>` | 248069 |

The Qwen3.8 tokenizer artifact is not byte-identical to Qwen3.6-27B. Psionic
must bind tokenizer identity to the Qwen3.8 artifact rather than assuming that
matching vocabulary size and special ids imply tokenizer equivalence.

The chat template changes behavior relative to Qwen3.6:

- thinking is enabled by default
- `reasoning_effort` accepts `xhigh`, `medium`, and `low`; `xhigh` is the
  default
- `xhigh` and `low` inject explicit reasoning instructions into the system
  turn; `medium` does not add an instruction
- `preserve_thinking` defaults to true
- preserved assistant reasoning is read from `reasoning_content`
- thinking can be disabled with `enable_thinking = false`
- tools use the Qwen XML-like `<tool_call>`, `<function=...>`, and
  `<parameter=...>` framing
- adjacent tool results are grouped into a user-side tool-response turn
- images and videos are refused in system messages

These semantics require a distinct Qwen3.8 template contract and fixture even
if the decoder kernels are shared with Qwen3.6.

Psionic implements that frontend contract as `qwen3.8.chat_template.v1` and
binds it to the template and tokenizer digests above. The retained golden
fixture is `fixtures/qwen38/qwen38_prompt_tokenizer_golden_v1.json`. Its source
rows were rendered with Transformers 5.15.0 against the pinned local artifact.
The generic GGUF tokenizer now routes `qwen35` through the published regex and
NFC normalization rather than the generic three-digit numeric grouping.

## Generation Defaults

The published generation config enables sampling with:

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 20`
- EOS ids `248046` and `248044`
- pad id `248044`

The model card recommends different non-thinking defaults:
`temperature = 0.7`, `top_p = 0.8`, `top_k = 20`, and
`presence_penalty = 1.5`.

## Context Extension

The native context length is 262,144 tokens. The model card describes a
1,000,000-token YaRN configuration with factor 4.0 and
`original_max_position_embeddings = 262144`.

Psionic must report native and extended context separately. Static YaRN can
reduce short-context quality, and the upstream card recommends enabling it
only when longer contexts are required.

## Upstream Performance Claims

The model card reports gains over Qwen3.6-27B in coding, long-horizon agent
work, instruction following, and multimodal tool use. Selected reported rows
include:

| Benchmark | Qwen3.8-27B | Qwen3.6-27B |
| --- | ---: | ---: |
| Terminal Bench 2.1 | 73.0 | 63.4 |
| SWE-bench Pro | 61.7 | 53.5 |
| NL2Repo-Bench | 42.3 | 36.2 |
| DeepSWE 1.1 | 42.2 | 13.3 |
| QwenSWEBench | 79.0 | 49.3 |
| CoWorkBench | 70.7 | 61.0 |
| JobBench | 33.4 | 21.8 |
| IFBench | 79.5 | 69.1 |
| ClawEval-MM pass@3 | 57.4 | 42.6 |

These are upstream-reported results. They are not Psionic benchmark results
and do not establish any local capability claim.

## Source Snapshot

| File | SHA-256 observed on 2026-08-14 |
| --- | --- |
| `README.md` | `57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641` |
| `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `generation_config.json` | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `video_preprocessor_config.json` | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `model.safetensors.index.json` | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |

The `tokenizer.json` digest above is the SHA-256 of the downloaded LFS blob.
The raw Git representation is an LFS pointer and has a different digest.

## Local Acquisition

The full repository was downloaded on 2026-08-14 with Hugging Face CLI 1.27.0:

```bash
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir target/models/qwen/Qwen3.8-27B \
  --max-workers 4
```

Verification facts:

- local directory: `target/models/qwen/Qwen3.8-27B`
- repository files verified: 32
- weight shards: 18
- incomplete downloads: 0
- indexed tensors: 1,199
- indexed tensor bytes: 55,562,855,904
- complete shard-file bytes including safetensors headers: 55,563,006,776
- `hf cache verify` result: all checksums match

The local CLI metadata under `.cache/huggingface/` is expected download state
and is not an upstream repository file. Verification was run with missing-file
enforcement while allowing that local metadata.

## Primary Sources

- model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- repository API: <https://huggingface.co/api/models/Qwen/Qwen3.8-27B>
- config: <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json>
- tokenizer config:
  <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/tokenizer_config.json>
- template:
  <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja>
- safetensors index:
  <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json>
- image processor:
  <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/preprocessor_config.json>
- video processor:
  <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/video_preprocessor_config.json>
