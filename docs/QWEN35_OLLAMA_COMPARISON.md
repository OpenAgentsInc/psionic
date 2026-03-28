# Qwen3.5 Native CUDA vs Ollama

This document is the canonical comparison matrix for Psionic native CUDA
`qwen35` inference versus local Ollama on this host.

Tracked issues:

- `#606` Scale native qwen35 CUDA lane to beat Ollama on 2B, 4B, and 9B
- `#631` Benchmark qwen35 temperature/top-k sampling vs Ollama and keep Psionic ahead

Published benchmark checkpoints:

- greedy matrix checkpoint: `c5bc0ba2`
- sampled matrix checkpoint: `213f3995`

Shared benchmark rules:

- same host
- same GGUF artifact digests
- same prompt per benchmark mode
- same token cap
- Psionic uses the native CUDA `qwen35` lane
- Ollama uses the local `ollama serve` instance
- decode throughput is reported as mean `tok/s`
- benchmark one runtime at a time for `9b` on this 16 GB RTX 4080 because
  Ollama keeps model weights resident in VRAM

## Greedy Contract

Prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

## Greedy Matrix

| Model | Artifact path | Artifact digest | Psionic decode tok/s | Ollama decode tok/s | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `qwen3.5:0.8b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf` | `afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5` | `523.20` | `328.72` | `implemented_early`, ahead | Current pushed checkpoint `c5bc0ba2` |
| `qwen3.5:2b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf` | `b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297` | `244.03` | `205.24` | `implemented_early`, ahead | Fresh March 27 rerun from a no-incremental rebuilt `qwen35_cuda_bench` example and a serialized local Ollama warmup pass |
| `qwen3.5:4b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf` | `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` | `166.75` | `141.62` | `implemented_early`, ahead | Mixed `Q4_K` and `Q6_K` row. The win required fixing the fused decode output head to use `Q8_1` projection plus `argmax_f32` for `Q6_K` output weights |
| `qwen3.5:9b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf` | `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` | `102.68` | `94.62` | `implemented_early`, ahead | The row fits and runs natively on this host once the local Ollama GPU caches are unloaded before the Psionic measurement |

## Sampled Contract

Prompt:

```text
Respond with plain text only. Write 20 numbered one-sentence reasons why GPU decode throughput matters for local inference. Start at 1 and keep going until all 20 are written.
```

Token cap:

- `128`

Sampled settings:

- `temperature = 0.8`
- `top_k = 40`
- `top_p = 0.9`
- `repeat_penalty = 1.0`
- `presence_penalty = 0.0`
- `frequency_penalty = 0.0`
- `seed = 42`
- `think = false` on Ollama

Psionic output-mode evidence on this contract:

- `qwen35_output_modes=[top_k_candidates:40]`
- `qwen35_raw_logits=false`
- Psionic sampled rows rerun after a clean `CARGO_INCREMENTAL=0` rebuild of
  `qwen35_cuda_bench`

## Sampled Matrix

| Model | Artifact path | Artifact digest | Psionic decode tok/s | Ollama decode tok/s | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `qwen3.5:0.8b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf` | `afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5` | `497.09` | `330.83` | `implemented_early`, ahead | Native CUDA sampled decode stays on the bounded `top_k_candidates` lane instead of dense vocab-logit readback |
| `qwen3.5:2b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf` | `b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297` | `244.07` | `202.41` | `implemented_early`, ahead | One Psionic repeat stopped at `112` tokens before the cap and still stayed ahead on decode throughput |
| `qwen3.5:4b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf` | `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` | `172.89` | `139.79` | `implemented_early`, ahead | Mixed `Q4_K` and `Q6_K` row stays ahead on the same bounded sampled lane |
| `qwen3.5:9b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf` | `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` | `105.90` | `93.08` | `implemented_early`, ahead | The row still needs the same operational rule as greedy benchmarking: unload Ollama before the Psionic measurement |

## Current Notes

- The `0.8b`, `2b`, `4b`, and `9b` rows are ahead on decode throughput on this
  host under both the greedy and sampled contracts above.
- The sampled CUDA lane is bounded, not vague. It uses
  `TopKCandidates { top_k }` only when the request stays inside the exact
  envelope:
  - sampled decode or non-zero effective temperature
  - effective `top_k` available and `<= 128`
  - repeat, presence, and frequency penalties inactive
  - structured-output masking inactive
- Requests outside that envelope still fall back to explicit raw-logit readback
  instead of silently narrowing behavior.
- The 4B row only became correct and faster after fixing the fused
  `ArgmaxOnly` output path. The hot decode branch now routes `Q6_K` output
  weights through `Q8_1` projection plus `argmax_f32` instead of the slower
  generic quantized matvec path.
- The 9B row does not require a separate Psionic fallback path on this host.
  The only extra rule is operational: unload Ollama's resident GPU caches
  before measuring Psionic because Ollama keeps prior model weights live in
  VRAM.
