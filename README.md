# Psionic

Psionic is a Rust-native ML and inference stack.

It owns the machine-facing execution substrate behind local inference, serving,
training, distributed execution, artifact truth, and clustered compute. The
project is broader than one app or one benchmark lane. It is the crate family
that OpenAgents uses for inference, training, cluster bring-up, and execution
evidence.

## Start Here

- System architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Detailed workspace map: [docs/WORKSPACE_MAP.md](docs/WORKSPACE_MAP.md)
- Inference and serving: [docs/INFERENCE_ENGINE.md](docs/INFERENCE_ENGINE.md)
- Inference mesh ownership: [docs/INFERENCE_MESH_OWNERSHIP.md](docs/INFERENCE_MESH_OWNERSHIP.md)
- Optimizer substrate: [docs/OPTIMIZER_SUBSTRATE.md](docs/OPTIMIZER_SUBSTRATE.md)
- Forge-facing eval pack publication: [docs/PSION_FORGE_EVAL_PACK_MANIFESTS.md](docs/PSION_FORGE_EVAL_PACK_MANIFESTS.md)
- Hermes user guide: [docs/hermes/README.md](docs/hermes/README.md)
- Training system: [docs/TRAIN_SYSTEM.md](docs/TRAIN_SYSTEM.md)
- Repo-local library roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Psion learned-model program: [docs/PSION_PROGRAM_MAP.md](docs/PSION_PROGRAM_MAP.md)
- Psion local-first operator runbook: [docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md](docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md)

## Main Tracks

- Inference and local serving
  - local GPT-OSS server and benchmark harness
  - generic OpenAI-compatible server surfaces
  - hardware validation and backend truth
  - start with [docs/GPT_OSS_LOCAL_SERVING.md](docs/GPT_OSS_LOCAL_SERVING.md)
- Hermes agent backend
  - use Psionic as a real Hermes backend over the OpenAI-compatible
    `chat.completions` path
  - start with [docs/hermes/README.md](docs/hermes/README.md)
  - supporting docs: [docs/HERMES_QWEN35_COMPATIBILITY.md](docs/HERMES_QWEN35_COMPATIBILITY.md), [docs/HERMES_QWEN35_PARALLEL_ATTRIBUTION.md](docs/HERMES_QWEN35_PARALLEL_ATTRIBUTION.md), [docs/HERMES_BACKEND_BENCHMARK.md](docs/HERMES_BACKEND_BENCHMARK.md)
- Parameter Golf and distributed training
  - single-H100, distributed `8xH100`, submission, evidence, and score-path work
  - start with [docs/ROADMAP_PARAMETERGOLF.md](docs/ROADMAP_PARAMETERGOLF.md)
  - supporting docs: [docs/PARAMETER_GOLF_SINGLE_H100_TRAINER.md](docs/PARAMETER_GOLF_SINGLE_H100_TRAINER.md), [docs/PARAMETER_GOLF_DISTRIBUTED_8XH100.md](docs/PARAMETER_GOLF_DISTRIBUTED_8XH100.md), [docs/PARAMETER_GOLF_RUNPOD_8XH100_RUNBOOK.md](docs/PARAMETER_GOLF_RUNPOD_8XH100_RUNBOOK.md)
- Cluster, swarm, and cross-provider compute
  - local mixed-hardware swarm, Google dual-node swarm, cross-provider training contracts
  - start with [docs/ROADMAP_CLUSTER.md](docs/ROADMAP_CLUSTER.md)
  - supporting docs: [docs/INFERENCE_MESH_OWNERSHIP.md](docs/INFERENCE_MESH_OWNERSHIP.md), [docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md](docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md), [docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md](docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md), [docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md](docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md)
- Psion learned-model program
  - corpus, tokenizer, pretrain, trusted-cluster, and decentralized contribution work
  - start with [docs/PSION_PROGRAM_MAP.md](docs/PSION_PROGRAM_MAP.md)
  - supporting docs: [docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md](docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md), [docs/PSION_PRETRAIN_STAGE.md](docs/PSION_PRETRAIN_STAGE.md), [docs/PSION_TRUSTED_CLUSTER_RUN.md](docs/PSION_TRUSTED_CLUSTER_RUN.md), [docs/PSION_DECENTRALIZED_CONTRIBUTION.md](docs/PSION_DECENTRALIZED_CONTRIBUTION.md)

## Psion Training Shortcut

If you want the current top Psion training lane instead of guessing among
benchmark-adjacent lanes, run:

```bash
./TRAIN
```

That command now targets the canonical accelerator-backed Psion reference lane
on the admitted Tailnet CUDA host and writes the copied-back artifacts plus one
local operator manifest and summary under `~/scratch/psion_train_runs/<run_id>`.

Use:

```bash
./TRAIN --dry-run
./TRAIN --mode local_reference
```

for plan inspection and bounded CPU-reference fallback.

## Tassadar Training Shortcut

If you want the current default Tassadar training lane instead of guessing
among older bounded benchmark lanes, run:

```bash
./TRAIN_TASSADAR
```

That command now means the bounded trace-bound article-transformer
weight-production lane that produces the retained
`tassadar-article-transformer-trace-bound-trained-v0` family under
`fixtures/tassadar/runs/tassadar_article_transformer_weight_production_v1`.

The lane contract lives in
[docs/TASSADAR_DEFAULT_TRAIN_LANE.md](docs/TASSADAR_DEFAULT_TRAIN_LANE.md).

The operator launcher lives in
[docs/TASSADAR_TRAIN_LAUNCHER.md](docs/TASSADAR_TRAIN_LAUNCHER.md).

## Tassadar Executor Lane

Executor-class research and runtime work for exact computation starts with
[docs/ROADMAP_TASSADAR.md](docs/ROADMAP_TASSADAR.md).

## Local GPT-OSS Inference

Psionic ships a dedicated local GPT-OSS server in
`crates/psionic-serve/src/bin/psionic-gpt-oss-server.rs`. It exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Build it:

```bash
cargo build -p psionic-serve --bin psionic-gpt-oss-server --release
```

Run it on a Linux NVIDIA host:

```bash
./target/release/psionic-gpt-oss-server \
  -m /path/to/gpt-oss-20b-mxfp4.gguf \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096 \
  -ngl 999
```

Run it on Apple Silicon:

```bash
./target/release/psionic-gpt-oss-server \
  -m /path/to/gpt-oss-20b-mxfp4.gguf \
  --backend metal \
  --metal-mode native \
  --host 127.0.0.1 \
  --port 8080 \
  -c 1024 \
  -ngl 4
```

Call it:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gpt-oss-20b-mxfp4.gguf",
    "messages": [
      {"role": "system", "content": "You are ChatGPT."},
      {"role": "user", "content": "Why does HTTPS matter?"}
    ]
  }' | jq
```

Benchmark it against local `llama.cpp`:

```bash
scripts/benchmark-gpt-oss-vs-llama.sh \
  --psionic-backend cuda \
  --model /path/to/gpt-oss-20b-mxfp4.gguf \
  --llama-bin /path/to/llama-server \
  --json-out /tmp/psionic-gpt-oss-bench
```

More detail lives in [docs/GPT_OSS_LOCAL_SERVING.md](docs/GPT_OSS_LOCAL_SERVING.md).

## GPT-OSS Benchmark Proof

The current benchmark harness is `scripts/benchmark-gpt-oss-vs-llama.sh`. It
uses the explicit GPT-OSS system/developer/user request contract, checks visible
output equality, and records prompt-cache-hit throughput.

The closed benchmark proof referenced publicly here is:

- OpenAgents issue comment:
  [openagents#3248 comment 4028968842](https://github.com/OpenAgentsInc/openagents/issues/3248#issuecomment-4028968842)
- exact reported result on that host:
  - Psionic `prompt_cache_hit`: `172.84 tok/s`
  - `llama.cpp prompt_cache_hit`: `160.98 tok/s`
  - `prompt_cache_hit_visible_output_match=true`
  - visible output:
    `HTTPS protects users by encrypting traffic, preventing tampering, and confirming they are connected to the right website.`

That proof is grounded in the shipped server binary, the shipped benchmark
script, and the explicit hardware-validation posture in
[docs/HARDWARE_VALIDATION_MATRIX.md](docs/HARDWARE_VALIDATION_MATRIX.md).

## Project Shape

The main crate families are:

- framework core: `psionic-core`, `psionic-ir`, `psionic-compiler`, `psionic-runtime`
- backends: `psionic-backend-cpu`, `psionic-backend-cuda`, `psionic-backend-metal`
- serving and provider surfaces: `psionic-serve`, `psionic-provider`, `psionic-router`
- cluster and distributed execution: `psionic-cluster`, `psionic-collectives`, `psionic-distributed`, `psionic-net`
- training, eval, and optimizer substrate: `psionic-train`, `psionic-data`, `psionic-eval`, `psionic-adapters`, `psionic-optimize`

Use [docs/WORKSPACE_MAP.md](docs/WORKSPACE_MAP.md) for the full doc index,
crate map, and subsystem entrypoints.
