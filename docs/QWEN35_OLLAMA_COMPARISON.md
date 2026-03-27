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
| `qwen3.5:2b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf` | `b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297` | `247.21` | `203.23` | `implemented_early`, ahead | Fresh March 27 rerun from the current native CUDA head on the same `128` token decode benchmark |
| `qwen3.5:4b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf` | `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` | `166.61` | `141.36` | `implemented_early`, ahead | Mixed `Q4_K` and `Q6_K` row. The win required fixing the fused decode output head to use `Q8_1` projection plus `argmax_f32` for `Q6_K` output weights |
| `qwen3.5:9b` | pending | pending | pending | pending | pending | pull, harvest, benchmark, optimize |

## Current Notes

- The `0.8b`, `2b`, and `4b` rows are now ahead on decode throughput on this
  host under the same prompt and token-cap contract.
- The 4B row only became correct and faster after fixing the fused `ArgmaxOnly`
  output path. The hot decode branch now routes `Q6_K` output weights through a
  `Q8_1` projection plus `argmax_f32` instead of the slower generic quantized
  matvec path.
- The next delivery bar is to harvest and benchmark `9b`, then push the native
  CUDA lane further on `2b`, `4b`, and `9b`.
