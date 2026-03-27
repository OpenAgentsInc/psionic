# Qwen3.5 Native CUDA vs Ollama

This document is the canonical comparison matrix for Psionic native CUDA
`qwen35` inference versus local Ollama on this host.

Tracked issue:

- `#606` Scale native qwen35 CUDA lane to beat Ollama on 2B, 4B, and 9B

Benchmark contract:

- same host
- same prompt
- same token cap
- Psionic uses the native CUDA `qwen35` lane
- Ollama uses the local `ollama serve` instance
- decode throughput is reported as mean `tok/s`
- on this 16 GB RTX 4080, benchmark one runtime at a time for `9b` because
  Ollama keeps model weights resident in VRAM

Benchmark prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

## Matrix

| Model | Artifact path | Artifact digest | Psionic decode tok/s | Ollama decode tok/s | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `qwen3.5:0.8b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf` | `afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5` | `523.20` | `328.72` | `implemented_early`, ahead | Current pushed checkpoint `c5bc0ba2` |
| `qwen3.5:2b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf` | `b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297` | `244.03` | `205.24` | `implemented_early`, ahead | Fresh March 27 rerun from a no-incremental rebuilt `qwen35_cuda_bench` example and a serialized local Ollama warmup pass |
| `qwen3.5:4b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf` | `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` | `166.75` | `141.62` | `implemented_early`, ahead | Mixed `Q4_K` and `Q6_K` row. The win required fixing the fused decode output head to use `Q8_1` projection plus `argmax_f32` for `Q6_K` output weights |
| `qwen3.5:9b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf` | `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` | `102.68` | `94.62` | `implemented_early`, ahead | The row fits and runs natively on this host once the local Ollama GPU caches are unloaded before the Psionic measurement |

## Current Notes

- The `0.8b`, `2b`, `4b`, and `9b` rows are now ahead on decode throughput on
  this host under the same prompt and token-cap contract.
- The 4B row only became correct and faster after fixing the fused `ArgmaxOnly`
  output path. The hot decode branch now routes `Q6_K` output weights through a
  `Q8_1` projection plus `argmax_f32` instead of the slower generic quantized
  matvec path.
- The 9B row does not require a separate Psionic fallback path on this host.
  The earlier load failure came from live Ollama GPU residency, not from a
  native Psionic inability to admit the artifact.
- The next delivery bar is to push the native CUDA lane further on `2b`, `4b`,
  and `9b` from this now-complete comparison baseline.
