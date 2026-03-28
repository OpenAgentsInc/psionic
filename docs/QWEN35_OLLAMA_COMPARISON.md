# Qwen3.5 Native CUDA vs Ollama

This document is the canonical comparison matrix for Psionic native CUDA
`qwen35` inference versus local Ollama on this host.

Tracked issues:

- `#606` Scale native qwen35 CUDA lane to beat Ollama on 2B, 4B, and 9B
- `#631` Benchmark qwen35 temperature/top-k sampling vs Ollama and keep Psionic ahead
- `#632` Use bounded candidate decode for qwen35 temperature and top-k sampling

Published benchmark checkpoints:

- greedy matrix checkpoint: historical `c5bc0ba2`, rerun pending under the
  corrected explicit Ollama greedy contract
- sampled matrix checkpoint: `March 27, 2026 rerun after sampler-surface refresh`
- large-`top_k` sampled matrix checkpoint: `March 28, 2026 clean-host partitioned top-k rerun`

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
- the local Ollama `qwen3.5` path on this checkout routes through
  `runner/ollamarunner` and its active sampler surface is the one built by
  `sample.NewSampler(temperature, topK, topP, minP, seed, grammar)`

## Machine-Readable Evidence Contract

Michel's laptop run exposed a gap in the repo-owned evidence path, not in the
human-readable comparison doc alone.

The canonical repo-owned rerun path is now:

- per-row benchmark reports from
  `cargo run --release -p psionic-serve --example qwen35_cuda_bench -- ... --json-out report.json`
- sequential multi-row collection from
  `scripts/release/run-qwen35-ollama-matrix.sh`

Those artifacts preserve the per-run fields that matter for honest comparison:

- `prompt_tokens`
- `output_tokens`
- `prompt_s`
- `decode_s`
- `total_s`
- `decode_tok_s`
- `qwen35_output_modes`
- `qwen35_readback_bytes`
- `qwen35_raw_logits`
- rendered prompt and stop sequences
- effective sampled contract values
- host label, GPU memory, power-limit metadata, Psionic commit, and Ollama
  version at the matrix level

Constrained-host follow-up reruns should use that path instead of publishing a
reduced matrix that keeps only token count and mean throughput.

## Greedy Contract

Prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

Ollama greedy settings:

- `temperature = 0.0`
- `top_k = 1`
- `top_p = 1.0`
- `min_p = 0.0`
- `repeat_penalty = 1.0`
- `repeat_last_n = 0`
- `presence_penalty = 0.0`
- `frequency_penalty = 0.0`
- `seed = 42`
- `keep_alive = 0`

Historical note:

- the older March 27 greedy checkpoint is not current canonical evidence
- the old harness omitted these explicit Ollama greedy settings and therefore
  let Ollama use its default sampler surface instead of a forced greedy path
- the repo now forces the explicit no-sampling, no-penalty Ollama contract
  above, and the greedy matrix must be rerun under that contract before greedy
  throughput claims are treated as canonical again

## Greedy Matrix

Greedy rerun pending under the corrected explicit Ollama greedy contract above.

Historical checkpoint values from `c5bc0ba2` are intentionally not reproduced
here as canonical matrix truth because the older harness did not force Ollama
onto a real greedy path.

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
| `qwen3.5:0.8b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf` | `afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5` | `496.28` | `329.38` | `implemented_early`, ahead | Native CUDA sampled decode still stays on the bounded `top_k_candidates` lane instead of dense vocab-logit readback |
| `qwen3.5:2b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf` | `b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297` | `243.81` | `202.58` | `implemented_early`, ahead | Fresh March 28 rerun after the sampled request-surface refresh |
| `qwen3.5:4b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf` | `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` | `173.44` | `139.90` | `implemented_early`, ahead | Mixed `Q4_K` and `Q6_K` row stays ahead on the same bounded sampled lane |
| `qwen3.5:9b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf` | `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` | `105.25` | `92.90` | `implemented_early`, ahead | The row still needs the same operational rule as greedy benchmarking: unload Ollama before the Psionic measurement |

## Large-`top_k` Sampled Contract

Prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

Sampled settings:

- `temperature = 0.8`
- `top_k = 100`
- `top_p = 0.9`
- `min_p = 0.05`
- `seed = 42`
- `think = false` on Ollama

Psionic output-mode evidence on this contract:

- `qwen35_output_modes=[top_k_candidates:100]`
- `qwen35_raw_logits=false`
- clean-host rerun after the one-row partitioned CUDA top-k path replaced the
  slower radix-sort route for larger bounded candidate sets
- serialized Ollama residency between rows to keep the `4b` and `9b`
  measurements honest on this 16 GB RTX 4080

## Large-`top_k` Sampled Matrix

| Model | Artifact path | Artifact digest | Psionic decode tok/s | Ollama decode tok/s | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `qwen3.5:0.8b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf` | `afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5` | `416.06` | `320.24` | `implemented_early`, ahead | Large-candidate sampled decode now stays on the bounded candidate lane instead of falling off the older radix-sort path |
| `qwen3.5:2b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-2b-q8_0-registry.gguf` | `b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297` | `224.83` | `204.06` | `implemented_early`, ahead | The follow-on win comes from partitioning the one-row vocab scan across multiple CUDA blocks and merging back exactly |
| `qwen3.5:4b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-4b-q8_0-registry.gguf` | `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` | `163.12` | `124.38` | `implemented_early`, ahead | Mixed `Q4_K` and `Q6_K` row also stays ahead under the larger bounded-candidate contract |
| `qwen3.5:9b` | `/home/christopherdavid/models/qwen3.5/qwen3.5-9b-q4_k_m-registry.gguf` | `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` | `101.18` | `93.13` | `implemented_early`, ahead | The `9b` row still requires serialized Ollama residency, but the wider candidate set stays on the native bounded lane and remains ahead |

## Penalty-Active Psionic Follow-On

This is not a canonical Psionic-versus-Ollama matrix on this checkout, because
the local Ollama `qwen3.5` runner does not wire repeat, presence, and
frequency penalties through the same active sampler path. It is still a useful
runtime checkpoint for Psionic itself because these requests used to force
explicit dense `raw_logits` readback on qwen35 CUDA.

Prompt:

```text
Explain what Psionic is in one sentence.
```

Token cap:

- `128`

Sampled settings:

- `temperature = 0.8`
- `top_k = 40`
- `top_p = 0.9`
- `min_p = 0.05`
- `repeat_penalty = 1.1`
- `repeat_last_n = 64`
- `presence_penalty = 0.2`
- `frequency_penalty = 0.1`
- `seed = 42`

Psionic output-mode evidence on this contract:

- `qwen35_output_modes=[top_k_candidates:40]`
- `qwen35_raw_logits=false`

Psionic-only measured means:

| Model | Psionic decode tok/s | Notes |
| --- | ---: | --- |
| `qwen3.5:0.8b` | `89.29` | Same exact penalty-active request now stays on bounded candidates instead of dense readback |
| `qwen3.5:2b` | `120.71` | Clear throughput win over the earlier raw-logits penalty fallback on this host |
| `qwen3.5:4b` | `56.47` | The row stays on bounded candidates, but the short prompt remains noisier under penalties |
| `qwen3.5:9b` | `43.27` | The row also stays on bounded candidates without materializing dense logits on the host |

## Current Notes

- The `0.8b`, `2b`, `4b`, and `9b` rows are ahead on decode throughput on this
  host under the original sampled contract and the clean-host large-`top_k`
  sampled contract above.
- The greedy contract is currently bounded to "rerun pending under the
  corrected explicit Ollama greedy settings" rather than a live canonical
  throughput claim.
- The sampled CUDA lane is bounded, not vague. It uses
  `TopKCandidates { top_k }` only when the request stays inside the exact
  envelope:
  - sampled decode or non-zero effective temperature
  - effective `top_k` available and `<= 128`
  - structured-output masking inactive
  - `mirostat` inactive
- The runtime sampling surface now honors `min_p` and request-level
  `repeat_last_n` in addition to `temperature`, `top_k`, `top_p`, `min_p`,
  `typical_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`,
  `seed`, `mirostat`, `mirostat_tau`, and `mirostat_eta`.
- The generic OpenAI-compatible qwen35 server surface now forwards those same
  controls on `/v1/chat/completions` and `/v1/responses`.
- `repeat_last_n` follows the Ollama-compatible control contract in the local
  sampler and benchmark harness:
  - default `64`
  - `0` disables the penalty lookback window
  - `-1` expands the penalty window to the full available history
- `min_p` remains compatible with the bounded qwen35 CUDA sampled lane because
  Psionic applies it after exact top-k candidate selection on both the dense
  and bounded sampling paths.
- `typical_p`, repeat/presence/frequency penalties, `repeat_last_n`, and
  `mirostat` are supported on the Psionic runtime and request surfaces, but
  they are not part of the canonical Psionic-versus-Ollama matrix on this
  checkout because the local Ollama `qwen3.5` runner does not wire those
  controls through the same active sampler path.
- Native qwen35 structured outputs are also supported now. Greedy no-penalty
  structured requests stay on `TopKCandidates { top_k: 128 }` and use exact
  sparse allowed-logit gather on candidate misses, while structured requests
  outside that envelope still fall back to explicit dense `raw_logits`
  readback. Structured outputs are still a parity feature, not part of the
  bounded-candidate throughput matrix. The qwen35 proxy lane still refuses
  them. The tokenizer-side append cache now also buckets token ids by leading
  char so sparse schema recovery does not linearly rescan the full vocabulary
  when there is no candidate shortlist.
- The local `qwen35_cuda_bench` harness now reproduces native-versus-Ollama
  JSON object and JSON schema requests too through `--json-object` and
  `--json-schema-file`. It also writes machine-readable per-run reports through
  `--json-out`, and the repo-owned matrix collector now lives at
  `scripts/release/run-qwen35-ollama-matrix.sh`. The native qwen35 structured
  path is now replay-safe and can stay off dense raw-logit replay on the
  bounded greedy schema lane.
- Structured-output throughput is still outside the canonical matrix. After the
  leading-char token-cache bucketing pass, the matcher memo-path allocation
  cut, and a clean isolated rebuild on March 28, 2026, the local
  `qwen3.5:0.8b` summary-schema spot check measured native Psionic at about
  `78 tok/s` on the first bounded sparse-gather run and about `162 tok/s` mean
  across a warmed three-repeat pass, versus local Ollama at about `331 tok/s`.
  Psionic published
  `qwen35_output_modes=[top_k_candidates:128,sparse_logits:2,sparse_logits:3,sparse_logits:10]`,
  `qwen35_readback_bytes=5700`, and `qwen35_raw_logits=false` on the sparse
  run. The later warmed repeats stayed on `qwen35_output_modes=[top_k_candidates:128]`
  and hit the token cap without materializing `structured_output_value`, so
  this stays a parity note instead of a canonical throughput row.
- `mirostat` therefore remains a Psionic-side capability note, not a canonical
  beat-Ollama throughput claim.
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
