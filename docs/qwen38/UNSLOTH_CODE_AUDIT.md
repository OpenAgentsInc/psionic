# Unsloth Qwen3.8 and GGUF Code Audit

> Status: `planned` on 2026-08-14. This audit records reusable implementation
> facts from a pinned local Unsloth checkout. It does not establish Qwen3.8
> support in Psionic.

## Scope

The audited checkout is:

- repository: `unslothai/unsloth`
- local path: `/home/christopherdavid/code/unsloth`
- revision: `ba466ca095c53cc4139179580c5407ae6b22b48e`
- branch state at audit time: clean `main`
- core license: Apache-2.0
- `studio/` license: AGPL-3.0-only

All source links below are pinned to the audited revision. The separate Studio
license is an implementation boundary. Psionic can use the observed behavior
as research input, but must not copy Studio implementation into this
Apache-2.0 repository.

The audit covers:

- how Unsloth recognizes the Qwen3.5 architecture used by Qwen3.8
- how Unsloth delegates GGUF execution
- how Studio discovers, downloads, validates, and inspects GGUF artifacts
- how Studio estimates hybrid-model memory and selects a context size
- how prompt-template capabilities and token counts are handled
- which parts are absent and therefore cannot be adapted from this checkout

## Findings

### There is no Qwen3.8-specific loader

The audited checkout contains no Qwen3.8 model implementation. Qwen3.8 enters
the generic Qwen3.5 path because the official checkpoint declares:

```text
architectures = ["Qwen3_5ForConditionalGeneration"]
model_type = "qwen3_5"
text_config.model_type = "qwen3_5_text"
```

Unsloth recognizes `qwen3_5` as a flash-linear-attention family and uses its
generic Transformers loader and compiler path. It also forces the family onto
its float32 fallback list because float16 training gradients are known to
produce NaNs in that environment.

Relevant sources:

- [`unsloth/models/loader.py`, float32 fallback](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/models/loader.py#L120-L141)
- [`unsloth/models/loader.py`, FLA family registration](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/models/loader.py#L266-L317)
- [`unsloth/models/loader.py`, generic `AutoConfig` admission](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/models/loader.py#L661-L704)
- [`unsloth/models/loader.py`, generic compiled model path](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/models/loader.py#L1560-L1859)
- [`unsloth/models/loader.py`, text/VLM dispatch](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/models/loader.py#L1865-L1984)

Psionic should preserve two identities:

- the served artifact identity is Qwen3.8-27B
- the executable architecture identity is `qwen3_5` in Transformers and
  `qwen35` in GGUF

Adding a fictional `qwen38` architecture would make the loader disagree with
both the official checkpoint and the GGUF artifact.

### The Transformers path depends on external implementation packages

Unsloth does not locally define the complete Qwen3.5/Qwen3.8 graph. Its loader
depends on Transformers and `unsloth_zoo`, including a bundled
flash-linear-attention implementation. The audited package constraints include
`transformers <= 5.5.0` and `unsloth_zoo >= 2026.8.12`.

This is useful compatibility evidence, but it is not a Psionic implementation
source. Psionic needs explicit graph, state, tensor-layout, and backend
contracts inside its own crate boundaries.

### Qwen3.5 save remapping also applies to Qwen3.8

Unsloth detects Qwen3.5 VLM-form checkpoints during save and remaps state-dict
prefixes to the layout expected by Transformers:

```text
language_model.model.*      -> model.language_model.*
visual.*                    -> model.visual.*
language_model.lm_head.*    -> lm_head.*
```

Relevant sources:

- [`unsloth/save.py`, Qwen3.5 detection and remap](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/save.py#L717-L765)
- [`unsloth/save.py`, remap application](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/save.py#L6344-L6355)

This matters for a future safetensors export or training lane. It is not part
of the first GGUF execution target.

## GGUF Execution Boundary

Unsloth Studio does not execute GGUF graphs itself. Its GGUF backend manages a
`llama-server` subprocess:

- `load_model` starts `llama-server`
- generation proxies `/v1/chat/completions`
- `unload_model` terminates the subprocess

Source: [`studio/backend/core/inference/llama_cpp.py`, backend contract](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L3568-L3575).

The installer fetches prebuilt binaries from `unslothai/llama.cpp` and records
`ggml-org/llama.cpp` as the upstream project. It defaults to a moving `latest`
release unless a version is supplied.

Source: [`studio/install_llama_prebuilt.py`, binary source and version](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/install_llama_prebuilt.py#L250-L262).

Psionic must not describe this route as native execution. `llama.cpp` can be a
version-pinned comparator and a temporary bring-up oracle. Psionic support
requires Psionic to load the artifact, construct the graph, own state, execute
backend operations, and emit its own receipts.

## Artifact Discovery and Download

### Variant planning is exact

Studio turns Hub metadata into a concrete download plan. The plan contains:

- exact target paths
- exact byte sizes
- source digests when the Hub exposes them
- every shard required by a split quantization
- separately classified companion artifacts such as `mmproj` and MTP files

Sources:

- [`studio/backend/core/models/gguf_plan.py`, artifact model](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/models/gguf_plan.py#L22-L34)
- [`studio/backend/core/models/gguf_plan.py`, shard and companion grouping](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/models/gguf_plan.py#L217-L287)
- [`studio/backend/core/models/gguf_plan.py`, resolved plan](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/models/gguf_plan.py#L338-L382)

The download path resolves Hub metadata, checks available disk space, limits
the transfer to the planned files, uses one worker, writes a manifest, and can
resume from that manifest while offline.

Source: [`studio/backend/core/models/hf_download.py`, planned download and verification](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/models/hf_download.py#L605-L732).

The Studio quantization preference list places `UD-Q3_K_XL` near the front of
its supported choices. This corroborates that the selected target is a normal
Unsloth deployment artifact. It does not prove quality or residency on this
device.

Source: [`studio/backend/hub/utils/gguf.py`, quantization preference order](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/hub/utils/gguf.py#L30-L67).

### Psionic adaptation

The first artifact resolver should produce a deterministic artifact manifest
before load. It should contain:

- Hub repository and immutable revision
- requested quantization label
- every selected shard path in order
- expected and observed byte size
- expected and observed digest
- companion classification
- local materialization path

The first text-only target must explicitly exclude `mmproj` rather than
silently downloading or ignoring it. MTP weights need a separate disposition:
skip them for standard generation or load them only when the MTP execution
lane is requested.

## GGUF Admission

Studio reads GGUF key/value metadata without loading tensor payloads. It
handles ranged reads and split files, walks all metadata entries, and only
marks the parse complete after the declared key/value count has been consumed.

The parser records:

- GGUF magic and split index
- `general.architecture`
- chat template and tokenizer size
- context length and block count
- expert counts and leading dense blocks
- query and KV head counts
- embedding and feed-forward dimensions
- key and value head dimensions
- sliding-window and full-attention interval
- recurrent/SSM dimensions
- KDA dimensions
- NextN/MTP layer count

Source: [`studio/backend/core/inference/llama_cpp.py`, GGUF metadata parser](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L8479-L8769).

Psionic should adapt the field coverage and completion rule. The parser must
remain a structured GGUF reader. Architecture and tensor admission must not be
based on the filename or quantization label.

The admission receipt should report:

- `general.architecture == qwen35`
- every required Qwen3.5 hybrid metadata key
- the exact tensor inventory, dimensions, and GGML storage types
- full-attention versus recurrent layer topology
- tokenizer and template digests
- whether MTP tensors are present and whether they were loaded
- explicit refusal reasons for missing keys, types, dimensions, or shards

## Memory and Context Planning

Studio separates several memory components:

- model weights
- attention KV cache
- recurrent convolution and delta-net state
- MTP memory
- compute buffers and CUDA context reserve

Its estimator has architecture-specific KV paths and a recurrent-state
formula. It uses a 320 MiB CUDA context reserve, a 1.15 safety factor, and a
context-dependent scratch estimate. It then binary-searches a context size,
aligned down to 256 tokens, against a default 97 percent occupancy target.

Sources:

- [`studio/backend/core/inference/llama_cpp.py`, KV and recurrent-state estimation](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L7338-L7586)
- [`studio/backend/core/inference/llama_cpp.py`, compute and CUDA reserve](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L7757-L7915)
- [`studio/backend/core/inference/llama_cpp.py`, context fitting](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L7992-L8102)

The Qwen hybrid KV path only counts layers selected by the full-attention
interval. The current Studio branch does not add recurrent state in that same
path. Psionic must not reproduce that omission. A Qwen3.8 estimate must always
include:

```text
resident bytes = weights
               + full-attention KV
               + recurrent convolution state
               + recurrent delta-net state
               + optional rollback snapshots
               + optional MTP state and weights
               + execution scratch
               + backend reserve
```

Context fitting is an admission estimate, not proof. The runtime receipt must
record actual allocations, selected context, placement, and peak memory. A
request that does not fit must reduce context or refuse before generation; it
must not silently change quantization, artifact, or execution backend.

## Launch and Runtime Verification

Studio builds a `llama-server` command with the model path, port, parallelism,
context size, flash-attention policy, context-shift policy, and GPU placement.
It distinguishes manual GPU-layer placement, automatic `--fit`, and proven
full offload. It can pass a model-provided chat template through a temporary
Jinja file and attaches `mmproj` only for a vision lane.

Sources:

- [`studio/backend/core/inference/llama_cpp.py`, command construction](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L14879-L15017)
- [`studio/backend/core/inference/llama_cpp.py`, template and multimodal flags](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L15312-L15386)
- [`studio/backend/core/inference/llama_cpp.py`, fit integrity flags](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L18971-L19020)

After launch, Studio queries `/props` to verify effective server properties.
Psionic should use the same principle without inheriting the subprocess
boundary: planned values and effective runtime values belong in one receipt,
and disagreement is a refusal or qualification failure.

## Prompt, Thinking, and Tool Handling

Studio scans the embedded chat template for markers such as
`enable_thinking`, `reasoning_effort`, `preserve_thinking`, and tool syntax. It
uses those observations to expose template capabilities. It sends tools and
per-request template arguments through the server. For token accounting it
uses `llama-server`'s `/apply-template` and `/tokenize` endpoints rather than
estimating from message strings.

Sources:

- [`studio/backend/core/inference/llama_cpp.py`, template capability scan](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L1050-L1159)
- [`studio/backend/core/inference/llama_cpp.py`, rendered prompt token count](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/studio/backend/core/inference/llama_cpp.py#L21475-L21516)

Psionic should adapt rendered-prompt token accounting and explicit capability
reporting. It should not use substring detection as the authority for template
behavior. The pinned template digest and golden render cases must define the
contract for:

- thinking on and off
- `reasoning_effort` values `low`, `medium`, and `xhigh`
- `preserve_thinking`
- system, developer, user, assistant, and tool messages
- tool declarations, calls, and results
- multimodal placeholders, even though execution remains refused in the first
  lane

## Dynamic Quantization Is Not Implemented Here

The name `UD-Q3_K_XL` describes an Unsloth dynamic quantization profile, not a
single GGML storage type. The audited Unsloth checkout recognizes variant
names, but it does not contain the tensor-by-tensor Dynamic V3 selection policy.

The save path delegates Hugging Face-to-GGUF conversion and GGUF quantization
to `unsloth_zoo` and llama.cpp. IQ profiles require an importance matrix.

Sources:

- [`unsloth/save.py`, IQ importance-matrix requirement](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/save.py#L174-L187)
- [`unsloth/save.py`, conversion and quantization delegation](https://github.com/unslothai/unsloth/blob/ba466ca095c53cc4139179580c5407ae6b22b48e/unsloth/save.py#L2023-L2150)

Psionic does not need the profile-generation algorithm to execute a completed
GGUF. It does need support for every concrete GGML type and tensor layout found
inside the selected artifact. That inventory can only be finalized after the
exact pinned target is downloaded and parsed.

## Adaptation Decisions

| Unsloth behavior | Psionic decision | Reason |
|---|---|---|
| Reuse Qwen3.5 architecture identity | Adapt | Matches official config and GGUF architecture truth |
| Generic Transformers graph dependency | Do not adapt | Psionic needs native graph and backend ownership |
| `llama-server` subprocess for GGUF | Comparator only | Does not satisfy native Psionic execution |
| Exact Hub download plan and manifest | Adapt | Makes artifact identity and shard completeness deterministic |
| Automatic companion discovery | Adapt with lane policy | Text must exclude `mmproj`; MTP must be explicit |
| Header-only GGUF metadata admission | Adapt | Supports early deterministic refusal |
| Hybrid memory model | Adapt and correct | Must always add recurrent state to hybrid KV |
| Binary-search context fitting | Adapt as preflight | Runtime allocation remains the source of truth |
| `/props` effective-value verification | Adapt concept | Planned and effective values must agree in receipts |
| Template substring capability detection | Do not use as authority | Golden renders and template digest are deterministic |
| Render then tokenize for prompt length | Adapt | Counts the actual model input |
| Studio source implementation | Do not copy | `studio/` is AGPL-3.0-only |

## Roadmap Consequences

This audit adds the following requirements to the implementation roadmap:

1. Resolve and verify an exact GGUF artifact set before loader admission.
2. Parse all GGUF metadata and tensor descriptors before allocating weights.
3. Preserve `Qwen3.8-27B` artifact identity while executing `qwen35` graph
   semantics.
4. Account for attention KV, recurrent state, scratch, reserve, and optional
   MTP independently.
5. Treat `UD-Q3_K_XL` as a mixed profile and qualify every contained GGML type.
6. Skip MTP tensors for the first standard-generation lane and report that
   decision.
7. Render the pinned template before token counting and validate every Qwen3.8
   reasoning mode with golden cases.
8. Keep `llama.cpp` pinned as a comparator. Do not route a native support claim
   through a subprocess.

## Unresolved Inputs

The Unsloth checkout does not provide:

- the Dynamic V3 tensor-selection policy used to produce the target artifact
- a native GGUF execution implementation
- proof that the selected artifact fits this RTX 4080 at either planned
  context size
- Qwen3.8-specific output parity evidence
- Psionic-compatible code for Studio-only features

The exact artifact inventory, llama.cpp implementation, and measured local run
remain separate qualification inputs.
