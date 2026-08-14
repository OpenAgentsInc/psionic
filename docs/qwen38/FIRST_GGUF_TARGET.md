# Qwen3.8 First GGUF Target

> Status: `planned` on 2026-08-14. This document selects the first GGUF
> artifact for qualification. It does not claim that Psionic can execute the
> artifact yet.

## Decision

The first Psionic Qwen3.8 execution target is:

| Field | Value |
| --- | --- |
| Source repository | `unsloth/Qwen3.8-27B-GGUF` |
| Observed repository revision | `fdd03b8bbd279c1694563650e79d85a2373d9934` |
| Artifact | `Qwen3.8-27B-UD-Q3_K_XL.gguf` |
| Published rounded size | 13.4 GB |
| Quantization family | Unsloth Dynamic `UD-Q3_K_XL` |
| Initial modality | Text only |
| Initial backend | Native CPU, then native CUDA |
| Initial CUDA context gate | 4,096 tokens |
| Next CUDA context gate | 8,192 tokens |

The official `Qwen/Qwen3.8-27B` BF16 checkpoint remains the model,
architecture, tokenizer, and tensor-table authority. The GGUF is the first
bounded execution artifact.

## Why This Artifact

The local CUDA target is an NVIDIA GeForce RTX 4080 with 16,376 MiB total
VRAM. On 2026-08-14, the interactive host reported 15,257 MiB free before a
model load. The repository's rounded 13.4 GB artifact size corresponds to
about 12.5 GiB and leaves about 2.4 GiB of the observed free VRAM before
runtime allocations.

That remaining budget must cover:

- KV cache for the full-attention layers
- Gated DeltaNet recurrent state
- CUDA scratch and temporary buffers
- graph capture and replay allocations
- allocator fragmentation and safety margin

`UD-Q3_K_XL` is the best first target because it combines a useful residency
margin with Unsloth's mixed Dynamic V3 quantization. The upstream repository
describes Dynamic V3 as a preview. Psionic must measure its output quality and
must not repeat the upstream quality claim as a Psionic result without local
evidence.

## Candidate Ranking

| Artifact | Published size | Role | Decision |
| --- | ---: | --- | --- |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13.4 GB | Primary execution target | Qualify first |
| `Qwen3.8-27B-Q3_K_M.gguf` | 13.8 GB | Standard K-quant compatibility baseline | Qualify after the primary artifact |
| `Qwen3.8-27B-IQ4_XS.gguf` | 15.7 GB | Higher-bit comparison | Do not require full CUDA residency on this host |
| `Qwen3.8-27B-Q4_K_M.gguf` | 17.1 GB | Quality and explicit CPU-offload baseline | Comparator only on this host |

All listed artifacts can use the host's system RAM. That does not make them
equivalent CUDA targets. The first accelerated claim requires the primary
artifact's admitted weights and runtime state to remain on the GPU without
unreported host placement.

## Qualification Gates

Before implementation treats the primary artifact as admitted, retain:

- the immutable source revision
- the exact artifact byte size and SHA-256 after download
- the complete GGUF tensor-type inventory
- converter and source-model provenance present in the repository
- model, tokenizer, pre-tokenizer, and embedded-template metadata
- parity results against an upstream-supported reference runtime
- peak CPU and CUDA memory measurements for prefill and decode
- explicit placement evidence for every weight and runtime state allocation

The first CUDA acceptance run uses a 4,096-token context budget. An 8,192-token
envelope is accepted only after separate peak-memory and parity measurements.
Longer contexts require their own retained evidence. The model's declared
262,144-token native context is not a local 16 GiB CUDA support claim.

Reject the run before generation when:

- a required mixed-quantization tensor type is unsupported
- a tensor would silently dequantize to an unreported allocation
- the memory plan requires unreported host placement
- the admitted context cannot preserve the declared allocator margin
- artifact metadata or digests differ from the qualified descriptor

## Initial Exclusions

Do not load `mmproj-BF16.gguf` or `mmproj-F16.gguf` for the first text lane.
Each projector is approximately 930 MB and materially reduces the CUDA memory
margin. Native image and video support remains a separate roadmap milestone.

Multi-token prediction, 262,144-token operation, YaRN extension, Metal, and
training remain outside this artifact's first acceptance claim.

## Sources

- <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- <https://huggingface.co/Qwen/Qwen3.8-27B>
- <https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf>
