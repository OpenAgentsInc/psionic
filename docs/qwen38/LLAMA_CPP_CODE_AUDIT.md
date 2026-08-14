# llama.cpp Qwen3.8 Implementation Audit

> Status: `planned` on 2026-08-14. This audit records implementation facts
> from a pinned llama.cpp checkout and maps them to Psionic work. It does not
> establish Qwen3.8 support in Psionic.

## Scope

The audited checkout is:

- repository: `ggml-org/llama.cpp`
- local path: `/home/christopherdavid/code/llama.cpp`
- revision: `9b05354ec6fb58b4e665e9a39ebc40285c015638`
- branch state at audit time: clean `master`
- license: MIT

All upstream source links below are pinned to that revision. The audit covers
conversion, GGUF schema and loading, text graph construction, hybrid state,
CPU and CUDA operations, quantization, tokenization, chat templates, MTP,
vision conversion, and tests.

## Support Identity

The audited checkout contains no Qwen3.8 architecture symbol. It supports the
checkpoint through the existing Qwen3.5 architecture because the official
Qwen3.8 config still declares `Qwen3_5ForConditionalGeneration` and the
converted GGUF architecture is `qwen35`.

The dispatch path maps `LLM_ARCH_QWEN35` to `llama_model_qwen35`.

Sources:

- [`gguf-py/gguf/constants.py`, architecture name](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/gguf-py/gguf/constants.py#L1179-L1185)
- [`src/llama-model.cpp`, model dispatch](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-model.cpp#L307-L312)
- [`src/llama-arch.cpp`, hybrid classification](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-arch.cpp#L966-L984)

Psionic should preserve `qwen3.8-27b` as the product and served identity while
using the admitted `qwen35` GGUF graph. The absence of a Qwen3.8 string in
llama.cpp is not missing support.

## Conversion Contract

The GGUF is not a byte-for-byte repack of the official safetensors tensors.
The converter changes metadata, names, values, and head order to match the
runtime graph.

### Hybrid metadata mapping

The Qwen3.5 converter inherits Qwen3-Next hybrid metadata mapping:

| Hugging Face field | GGUF field |
|---|---|
| `linear_conv_kernel_dim` | `qwen35.ssm.conv_kernel` |
| `linear_key_head_dim` | `qwen35.ssm.state_size` |
| `linear_num_key_heads` | `qwen35.ssm.group_count` |
| `linear_num_value_heads` | `qwen35.ssm.time_step_rank` |
| `linear_value_head_dim * linear_num_value_heads` | `qwen35.ssm.inner_size` |
| `full_attention_interval`, default 4 | `qwen35.full_attention_interval` |

Source: [`conversion/qwen.py`, hybrid GGUF parameters](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/conversion/qwen.py#L364-L378).

### Value and name transformations

The inherited converter also:

- replaces `A_log` with `-exp(A_log)` before storage
- renames `dt_bias` to the GGUF SSM time-step bias
- removes the singleton dimension from convolution weights
- adds one to non-linear-attention norm weights
- splits fused per-head QKVZ projections into a QKV tensor and a separate gate
  tensor when that source layout is present

Source: [`conversion/qwen.py`, Qwen3-Next tensor transforms](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/conversion/qwen.py#L380-L420).

Psionic's runtime equations already expect the converted negative SSM A value:
it computes `exp(softplus(alpha + dt_bias) * A)`. Qualification must compare
the GGUF tensor semantics with the official safetensors transform instead of
comparing raw bytes or raw values.

### V-head reordering is mandatory

Qwen3.5 permits more value heads than key heads. Hugging Face stores value
heads grouped by key head. GGML broadcast expects them in tiled order:

```text
Hugging Face grouped: G0_v0, G0_v1, G1_v0, G1_v1
GGUF tiled:           G0_v0, G1_v0, G0_v1, G1_v1
```

The converter rewrites all affected axes:

- V rows inside the linear-attention QKV projection
- Z/gate rows
- alpha and beta rows
- `A_log`, time-step bias, and time-step projection entries
- V channels in the convolution kernel
- columns of the linear-attention output projection
- equivalent NVFP4 codes and scale groups

Sources:

- [`conversion/qwen.py`, reorder definition](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/conversion/qwen.py#L438-L477)
- [`conversion/qwen.py`, dense and quantized transforms](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/conversion/qwen.py#L479-L603)

The converter does not write a `qwen35.ssm.v_head_reordered` metadata key.
The layout follows from the `qwen35` converter and its version. Psionic
currently defaults its optional `qwen35.ssm.v_head_reordered` fact to `true`,
which matches this producer. That default is not sufficient provenance for an
arbitrary community conversion.

The Qwen3.8 qualification receipt must bind the artifact to a converter and
record `v_head_layout = tiled`. A `qwen35` artifact from an unknown producer
without layout evidence must refuse or pass a specific tensor-level parity
gate before execution.

### MRoPE sections

Qwen3.5 always uses interleaved MRoPE. If the source config omits its sections,
the converter writes `[11, 11, 10, 0]`.

Source: [`conversion/qwen.py`, Qwen3.5 MRoPE and class registration](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/conversion/qwen.py#L606-L629).

The loader requires four dimension-section entries. Psionic must admit the
artifact value, not inject the default after load.

### MTP conversion

The converter identifies appended MTP layers, maps their normalization and
projection tensors into decoder-layer slots after the trunk, and writes the
NextN layer count. It supports full conversion, trunk without MTP, and an
MTP-only artifact.

Source: [`conversion/qwen.py`, MTP tensor mapping](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/conversion/qwen.py#L300-L361).

The first Psionic generation lane should skip MTP tensors. It should still
inventory them and report the skip. MTP becomes a separate execution and
memory gate after base-model parity.

## GGUF Schema and Loading

The Qwen3.5 schema includes:

- token embedding, output norm, and optional output head
- attention norm and post-attention norm
- full-attention Q, K, V, Q/K norms, gate, and output tensors
- recurrent QKV, gate, convolution, A, time-step bias, alpha, beta, norm, and
  output tensors
- dense FFN gate, up, and down tensors
- optional NextN/MTP tensors

Source: [`gguf-py/gguf/constants.py`, Qwen3.5 tensor inventory](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/gguf-py/gguf/constants.py#L2672-L2703).

The model loader:

- requires RMS epsilon and four MRoPE sections
- loads SSM convolution, inner, state, time-step, and group dimensions
- loads an optional NextN layer count
- uses an explicit recurrent-layer array when present
- otherwise marks every fourth layer as full attention
- classifies a 64-layer dense trunk as 27B
- falls back to tied token embeddings if the output head is absent
- skips MTP tensors unless `load_mtp` is requested

Source: [`src/models/qwen35.cpp`, hyperparameters and tensor loading](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/models/qwen35.cpp#L4-L128).

Psionic should keep stricter artifact admission than the fallback loader:

- require the exact topology expected by the pinned Qwen3.8 config
- compare all 64 trunk layers with the declared full-attention interval
- inventory the appended MTP block separately
- report tied versus separate output weights
- refuse missing required Qwen3.8 fields instead of silently applying a
  family default

## Text Graph

### Trunk ordering

For each trunk layer llama.cpp executes:

1. pre-attention RMS norm
2. recurrent Gated DeltaNet or full attention
3. attention residual addition
4. post-attention RMS norm
5. dense gated SiLU FFN
6. FFN residual addition

After the trunk it applies the final RMS norm and LM head. MTP layers are not
executed in this main graph.

Source: [`src/models/qwen35.cpp`, main graph](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/models/qwen35.cpp#L137-L229).

### Full-attention block

The full-attention layer:

- projects queries and query gates jointly
- applies per-head RMS norm to Q and K
- applies partial interleaved MRoPE to Q and K
- runs scaled attention
- multiplies attention output by `sigmoid(query_gate)`
- applies the output projection

Source: [`src/models/qwen35.cpp`, full attention](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/models/qwen35.cpp#L258-L336).

### Recurrent block

The recurrent layer:

1. projects QKV, Z, alpha, and beta
2. computes `sigmoid(beta)`
3. computes `gate = softplus(alpha + dt_bias) * A`
4. prepends recurrent convolution state and runs causal convolution plus SiLU
5. splits convolved Q, K, and V
6. L2-normalizes Q and K with RMS epsilon
7. maps key heads to tiled value heads
8. updates the F32 delta-net state and computes the recurrent attention output
9. applies RMS norm gated by `SiLU(Z)`
10. applies the recurrent output projection

Source: [`src/models/qwen35.cpp`, recurrent attention](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/models/qwen35.cpp#L339-L470).

The CPU, CUDA, and comparator fixtures must validate intermediate values at
the convolution, decay/beta, normalized Q/K, delta state, gated norm, and
projected-output boundaries. End-token agreement alone will not localize a
layout error.

## Hybrid Memory and State

llama.cpp allocates a hybrid memory object for Qwen3.5:

- KV cache exists only for non-recurrent trunk layers
- convolution and delta-net state exist only for recurrent trunk layers
- both recurrent state tensors use F32
- state can be offloaded with the layer
- MTP uses a separate plain attention KV cache

Source: [`src/llama-model.cpp`, Qwen hybrid memory construction](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-model.cpp#L2252-L2334).

The recurrent cache allocates one convolution row and one delta-state row for
each recurrent layer, sequence, and optional rollback snapshot. It clears all
buffers to zero on allocation and supports full reset.

Sources:

- [`src/llama-memory-recurrent.cpp`, allocation and zeroing](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-memory-recurrent.cpp#L20-L128)
- [`src/llama-memory-recurrent.cpp`, clear and bounded rollback](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-memory-recurrent.cpp#L130-L232)
- [`src/llama-memory-hybrid.cpp`, hybrid batching and state operations](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-memory-hybrid.cpp#L67-L187)

For ordinary decode with one sequence and no rollback, recurrent state is
constant with context length. The formulas are:

```text
conv elements per recurrent layer =
    (conv_kernel - 1) * (inner_size + 2 * group_count * state_size)

delta elements per recurrent layer =
    state_size * inner_size

recurrent bytes =
    recurrent_layers * sequences * (1 + rollback_snapshots)
    * (conv_elements + delta_elements) * 4
```

Attention KV grows with context length and only covers full-attention layers.
The admission report must publish the two components separately. Reducing
context cannot remove the fixed recurrent-state cost.

Qwen3.5 supports bounded recurrent-state rollback in llama.cpp. This is
required for speculative verification, not for the first standard greedy or
sampled lane. Psionic should refuse MTP speculative decoding until it owns
equivalent rollback semantics and tests state restoration.

## CPU Operations

llama.cpp implements F32 SSM convolution and F32 Gated DeltaNet in the generic
CPU backend.

Sources:

- [`ggml/src/ggml-cpu/ops.cpp`, SSM convolution](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cpu/ops.cpp#L9555-L9623)
- [`ggml/src/ggml-cpu/ops.cpp`, Gated DeltaNet](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cpu/ops.cpp#L10744-L10954)

The delta state is stored transposed. The operator returns attention outputs
plus one or more updated state snapshots. Qwen3.8 CPU parity must cover both
prefill and token-at-a-time decode because those paths exercise different
state spans.

## CUDA Operations

The CUDA backend has native dispatch for both required recurrent operations.

### SSM convolution

The CUDA SSM convolution:

- operates in F32
- requires the channel count to be divisible by 128
- supports convolution widths 3, 4, 5, 9, and 15
- has separate short and long-token kernels
- can fuse channel bias and SiLU

Qwen3.8 uses width 4, which is inside this envelope.

Sources:

- [`ggml/src/ggml-cuda/ssm-conv.cu`, kernels and constraints](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cuda/ssm-conv.cu#L5-L205)
- [`ggml/src/ggml-cuda/ggml-cuda.cu`, supported-op check](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cuda/ggml-cuda.cu#L5208-L5211)

### Gated DeltaNet

The CUDA Gated DeltaNet:

- operates on F32 Q, K, V, gate, beta, and state
- specializes value-head dimensions 16, 32, 64, and 128
- keeps the transposed state in registers per output column
- supports ordinary scalar decay and KDA vector decay
- can emit rollback snapshots
- can fuse snapshot writes directly into the recurrent cache

Qwen3.8 uses value-head dimension 128, which is supported.

Sources:

- [`ggml/src/ggml-cuda/gated_delta_net.cu`, kernel and launch specialization](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cuda/gated_delta_net.cu#L4-L220)
- [`ggml/src/ggml-cuda/gated_delta_net.cu`, state and dispatch](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cuda/gated_delta_net.cu#L223-L327)
- [`ggml/src/ggml-cuda/ggml-cuda.cu`, snapshot-cache fusion](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cuda/ggml-cuda.cu#L2729-L2785)

Psionic already has Qwen3.5-specific CUDA operations. Reuse still requires
real Qwen3.8 intermediate parity and explicit prefill/decode tests. Shape
compatibility alone is insufficient.

## Quantization

### `UD-Q3_K_XL` is not a GGML type

The official llama.cpp checkout does not define an `UD-Q3_K_XL` quantization
type or profile. `UD-Q3_K_XL` is an Unsloth artifact recipe. The resulting
file contains ordinary GGML tensor type codes, potentially mixed per tensor.

llama.cpp loads the tensor types in the file. It does not need to understand
the filename label.

### Q3_K format

The selected filename strongly implies that Q3_K is one of its main storage
types. Q3_K stores 256 weights per 110-byte super-block, for 3.4375 effective
bits per weight. It contains low two-bit quants, a high-bit mask, 6-bit
sub-block scales, and one FP16 super-block scale.

Sources:

- [`ggml/src/ggml-common.h`, Q3_K block layout](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h#L311-L321)
- [`ggml/src/ggml-quants.c`, reference Q3_K dequantization](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-quants.c#L1305-L1353)
- [`ggml/src/ggml-cpu/ggml-cpu.c`, Q3_K CPU dot product registration](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cpu/ggml-cpu.c#L298-L314)
- [`ggml/src/ggml-cuda/mmvq.cu`, Q3_K CUDA matvec registration](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-cuda/mmvq.cu#L22-L51)

### Psionic R5 resolution

R5 resolves the original Q3_K blocker for the selected Qwen3.8 artifact set.
Psionic now maps the concrete GGUF tensor types found in the three
materialized files to native loader, CPU, and CUDA storage paths: `Q3_K`,
`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ3_S`, and `IQ4_XS`.

Relevant Psionic sources:

- `crates/psionic-models/src/lib.rs`, `GgufTensorType`, around lines
  1850-2000
- `crates/psionic-models/src/lib.rs`, GGML block decoders, around lines
  9600-9860
- `crates/psionic-backend-cpu/src/lib.rs`, GGML row dot/decode helpers
- `crates/psionic-backend-cuda/src/kernels/quantized_matvec.cu`, native CUDA
  super-block kernels

The R5 retained reports inspect the exact target artifacts and list every
required tensor type. Future community GGUFs still need the same inventory
gate before execution. The runtime must not create unreported dense F16
mirrors that defeat the 16 GiB residency plan.

## Tokenizer

llama.cpp assigns `tokenizer.ggml.pre = qwen35` to a dedicated pretokenizer.
Its regex differs from Qwen2 in two important ways:

- letter runs include Unicode combining marks, `\p{M}`
- each numeric code point is split independently with `\p{N}`, not grouped in
  runs of one to three digits

Sources:

- [`src/llama-vocab.cpp`, Qwen3.5 pretokenizer selection](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-vocab.cpp#L372-L388)
- [`src/unicode.cpp`, dedicated Qwen3.5 regex implementation](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/unicode.cpp#L608-L675)
- [`src/unicode.cpp`, regex dispatch](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/unicode.cpp#L1050-L1065)

R2 now routes `Qwen35` through the dedicated published regex and applies the
official tokenizer's NFC normalizer before byte-level BPE. The retained fixture
covers multi-digit strings, combining marks, contractions, punctuation, spaces,
and newlines.

An empirical `llama-tokenize` pass at the pinned revision against
`Qwen3.8-27B-UD-Q3_K_XL.gguf` matched eight of nine cases. The decomposed input
`café café` is the exception: the official tokenizer and Psionic emit
`[895, 56868, 50203]`, while llama.cpp emits
`[895, 56868, 39579, 52033]`. The GGUF-loaded llama.cpp path does not apply the
official NFC normalizer for that input. Psionic uses the official normalization
contract. The exact comparator revision, artifact, counts, and divergent IDs
are retained in
`fixtures/qwen38/qwen38_prompt_tokenizer_golden_v1.json`.

## Chat Template and Reasoning Controls

llama.cpp uses its generic Jinja engine for the embedded model template. The
server passes `chat_template_kwargs` into the template context. Its top-level
OpenAI-compatible `reasoning_effort` handling only treats `none` as disabling
thinking; other values do not affect the template automatically.

Sources:

- [`tools/server/server-common.cpp`, template arguments and reasoning effort](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/tools/server/server-common.cpp#L1278-L1306)
- [`common/chat.cpp`, Jinja extra context](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/common/chat.cpp#L3371-L3448)
- [`tools/server/README.md`, public parameter behavior](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/tools/server/README.md#L1249-L1255)

The Qwen3.8 template expects `reasoning_effort = low`, `medium`, or `xhigh`.
Comparator requests must place those values in `chat_template_kwargs` or use
`/apply-template` directly. Sending only top-level `reasoning_effort` would
produce a false parity result.

Psionic should keep its own digest-bound Qwen3.8 renderer and compare rendered
bytes and token IDs. The comparator is not the prompt authority.

## MTP

Qwen3.8 publishes one appended MTP block. llama.cpp:

- skips the MTP tensors during ordinary model load
- loads them only when draft-MTP is requested
- builds a separate dense-attention graph for the one appended block
- combines the prior target hidden row with the next-token embedding
- runs attention, FFN, norm, and an optional shared or main LM head
- uses a single trained head for Qwen3.5/Qwen3.8 speculative drafting

Sources:

- [`src/models/qwen35.cpp`, MTP load and graph](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/models/qwen35.cpp#L40-L126)
- [`src/models/qwen35.cpp`, MTP execution](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/models/qwen35.cpp#L488-L644)
- [`common/speculative.cpp`, Qwen3.5 single-head MTP mode](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/common/speculative.cpp#L1274-L1324)
- [`common/common.cpp`, conditional MTP load](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/common/common.cpp#L1686-L1700)

The first Qwen3.8 claim does not require MTP. Loading unused MTP weights would
consume residency without changing base-model output. A later MTP claim needs
separate quality, acceptance-rate, rollback, memory, and performance evidence.

## Vision Conversion

The same `Qwen3_5ForConditionalGeneration` class is registered with the
Qwen3-VL multimodal converter. It creates a separate projector artifact,
filters out text and MTP tensors, maps deep-stack and merger tensors, and
splits the temporal width-two Conv3D patch embedding into two Conv2D tensors.

Source: [`conversion/qwen3vl.py`, Qwen3.5 vision conversion](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/conversion/qwen3vl.py#L16-L143).

This confirms the current text-first split:

- main GGUF owns the `qwen35` text graph
- `mmproj` owns native vision preprocessing and projection
- prompt markers without `mmproj` do not provide vision execution

The selected text target must not auto-attach an `mmproj`. Vision remains its
own Psionic milestone.

## Multi-GPU Partitioning

llama.cpp has Qwen3.5-specific tensor segmentation because tiled V heads must
remain aligned with their key heads. It also constrains split granularity by
quantization block size, head dimension, and 128-element backend alignment.

Source: [`src/llama-model.cpp`, Qwen3.5 split segments and granularity](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/src/llama-model.cpp#L525-L669).

The first single-RTX-4080 lane does not need tensor parallelism. Future
multi-GPU work must preserve these head groups and recurrent-cache partitions;
generic row splitting is not safe.

## Upstream Test Coverage

The audited checkout includes:

- synthetic Qwen3.5 architecture generation
- Qwen3.5 tokenizer fixtures
- Qwen3.5 chat-template parser tests
- CPU/backend operation tests for realistic SSM convolution and Gated
  DeltaNet shapes
- recurrent-state rollback tests using a generated Qwen3.5 model

Sources:

- [`tests/CMakeLists.txt`, tokenizer and rollback tests](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/tests/CMakeLists.txt#L120-L145)
- [`tests/CMakeLists.txt`, generated-model rollback fixture](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/tests/CMakeLists.txt#L197-L219)
- [`tests/test-backend-ops.cpp`, Gated DeltaNet cases](https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/tests/test-backend-ops.cpp#L4328-L4380)

These tests validate the reusable implementation. They do not prove output
parity for the real Qwen3.8-27B artifact or the selected Dynamic V3 GGUF.
Psionic needs retained real-artifact evidence.

## Adaptation Decisions

| llama.cpp behavior | Psionic decision | Reason |
|---|---|---|
| Execute Qwen3.8 as `qwen35` | Adapt | Matches official config and GGUF truth |
| Converter value transforms | Validate explicitly | Raw safetensors and GGUF values are not identical |
| Tiled V-head conversion | Adapt and receipt-bind | Required for correct Q/K-to-V mapping |
| Default `[11,11,10,0]` MRoPE sections | Require from artifact | Avoid silent loader defaults |
| Every fourth layer full attention fallback | Do not rely on fallback | Pinned model topology should be exact |
| Hybrid KV plus F32 recurrent state | Adapt | Required for correct execution and memory truth |
| Recurrent rollback snapshots | Defer with MTP | Not needed for first standard generation lane |
| Q3_K CPU and CUDA operations | Adapt if present, expected required | Primary artifact likely uses Q3_K |
| Dedicated Qwen3.5 tokenizer regex | Adapt | Generic Psionic regex is not exact |
| Generic Jinja execution | Comparator only | Psionic keeps a digest-bound renderer |
| Conditional MTP load | Adapt | Avoid unused weight residency |
| Separate vision projector | Adapt later | Preserves text/vision claim boundary |
| Qwen3.5 multi-GPU segmentation | Defer | First lane is single GPU |

## Roadmap Consequences

This audit adds the following hard gates:

1. Implement exact Qwen3.5 pretokenization before prompt parity.
2. Bind the selected artifact to converter revision and tiled V-head layout.
3. Validate converter transforms against sampled official BF16 tensors.
4. Add a Q3_K runtime mode, block decoder, CPU projection, and CUDA projection
   when the target inventory confirms Q3_K.
5. Implement every other concrete tensor type in the Dynamic V3 artifact
   before load; do not infer support from the filename.
6. Publish attention KV and F32 recurrent-state memory independently.
7. Test prefill and decode state transitions, clean reset, and repeated
   requests on CPU and CUDA.
8. Skip and report MTP for the first claim. Treat speculative MTP as a
   follow-on capability with rollback tests.
9. Pin llama.cpp revision
   `9b05354ec6fb58b4e665e9a39ebc40285c015638` for comparator evidence.
10. Pass Qwen3.8 reasoning levels through template arguments during comparator
    tests so `low` and `medium` are actually rendered.

## Remaining Unknowns

This checkout does not answer:

- the exact mixed tensor-type inventory of
  `Qwen3.8-27B-UD-Q3_K_XL.gguf`
- the Unsloth Dynamic V3 tensor-selection recipe
- actual RTX 4080 residency at 4,096 or 8,192 tokens
- numerical parity between Psionic and the real artifact
- whether a future target revision changes conversion layout or template
  behavior

Those require the pinned artifact download, structured inspection, and
measured qualification runs.
