# Psionic Workspace Map

This document replaces the old oversized README inventory.

`README.md` is now the short front door. This file is the expanded map for the
main crate groups, canonical docs, and active subsystem lanes.

## Canonical Docs

Read these first:

- `README.md`
  - short project overview and quick entrypoints
- `docs/ARCHITECTURE.md`
  - canonical Psionic-wide system spec
- `docs/INFERENCE_ENGINE.md`
  - canonical inference and serving completion bar
- `docs/OPTIMIZER_SUBSTRATE.md`
  - canonical optimizer substrate spec for bounded offline search artifacts
- `docs/TRAIN_SYSTEM.md`
  - canonical training-system spec
- `docs/ROADMAP.md`
  - full-library roadmap

Useful lane-specific docs:

- `docs/ROADMAP_PARAMETERGOLF.md`
- `docs/ROADMAP_CLUSTER.md`
- `docs/ROADMAP_METAL.md`
- `docs/ROADMAP_MLX.md`
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
- `docs/HARDWARE_VALIDATION_MATRIX.md`

## Design Rules

- Keep manifests, receipts, capability reports, and artifact identity explicit.
- Keep refusal boundaries explicit. Do not hide unsupported paths behind silent
  fallback.
- Keep crate boundaries clean. App UX, payouts, and settlement logic do not
  belong here.
- Treat audits as rationale, not canonical current-state spec.

## Crate Groups

### Framework core

These crates own tensor, graph, runtime, and backend contracts:

- `crates/psionic-core`
- `crates/psionic-ir`
- `crates/psionic-compiler`
- `crates/psionic-runtime`

Reference:

- `docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md`
- `docs/BACKENDS.md`

### Backend execution

These crates own real backend behavior and backend-facing validation:

- `crates/psionic-backend-cpu`
- `crates/psionic-backend-cuda`
- `crates/psionic-backend-metal`
- `crates/psionic-backend-tests`

Reference:

- `docs/HARDWARE_VALIDATION_MATRIX.md`
- `docs/BACKENDS.md`

### Serving and model execution

These crates own local server surfaces, model execution, routing, and provider
projection:

- `crates/psionic-models`
- `crates/psionic-serve`
- `crates/psionic-provider`
- `crates/psionic-router`
- `crates/psionic-catalog`

Reference:

- `docs/INFERENCE_ENGINE.md`
- `docs/GPT_OSS_LOCAL_SERVING.md`
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
- `docs/NON_GPT_OSS_QWEN_PILOT.md`

### Cluster and distributed execution

These crates own topology, collectives, networking, sandboxing, and distributed
runtime truth:

- `crates/psionic-cluster`
- `crates/psionic-collectives`
- `crates/psionic-distributed`
- `crates/psionic-datastream`
- `crates/psionic-net`
- `crates/psionic-sandbox`

Reference:

- `docs/ROADMAP_CLUSTER.md`
- `docs/CLUSTER_VALIDATION_RUNBOOK.md`
- `docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md`
- `docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md`

### Training, eval, and adapters

These crates own training contracts, evaluation, datasets, adapter lanes, and
execution evidence:

- `crates/psionic-train`
- `crates/psionic-data`
- `crates/psionic-eval`
- `crates/psionic-adapters`
- `crates/psionic-optimize`
- `crates/psionic-nn-optimizers`

Reference:

- `docs/TRAIN_SYSTEM.md`
- `docs/OPTIMIZER_SUBSTRATE.md`
- `docs/ROADMAP_PARAMETERGOLF.md`
- `docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md`
- `docs/TRAIN_PROGRAM_MANIFEST_REFERENCE.md`

### Apple and MLX surfaces

These crates own Apple-side model and serving integration:

- `crates/psionic-apple-fm`
- `crates/psionic-mlx-compat`

Reference:

- `docs/ROADMAP_MLX.md`
- `docs/MLX_TEXT_SERVE.md`
- `docs/FM_API_COVERAGE_MATRIX.md`

## Major Efforts

### Inference and serving

Current entrypoints:

- local GPT-OSS server:
  `crates/psionic-serve/src/bin/psionic-gpt-oss-server.rs`
- benchmark harness:
  `scripts/benchmark-gpt-oss-vs-llama.sh`
- generic OpenAI-compatible server surface:
  `crates/psionic-serve/src/openai_http.rs`

Start with:

- `docs/GPT_OSS_LOCAL_SERVING.md`
- `docs/INFERENCE_ENGINE.md`
- `docs/HARDWARE_VALIDATION_MATRIX.md`

### Parameter Golf

Current focus:

- single-H100 trainer and evidence lane
- distributed `8xH100` runtime and runbook
- submission packaging, exported-folder evidence, and score-path closure

Start with:

- `docs/ROADMAP_PARAMETERGOLF.md`
- `docs/PARAMETER_GOLF_SINGLE_H100_TRAINER.md`
- `docs/PARAMETER_GOLF_DISTRIBUTED_8XH100.md`
- `docs/PARAMETER_GOLF_RUNPOD_8XH100_RUNBOOK.md`

### Cluster, swarm, and cross-provider training

Current focus:

- local first-swarm lane across Mac MLX and Linux CUDA
- Google two-node swarm lane
- cross-provider launch, binder, program-manifest, and evidence contracts

Start with:

- `docs/ROADMAP_CLUSTER.md`
- `docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md`
- `docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md`
- `docs/TRAIN_SYSTEM.md`

### Psion learned-model program

Current focus:

- corpus admission and tokenizer work
- pilot pretraining, checkpoint, observability, and acceptance docs
- trusted-cluster and decentralized contribution lanes

Start with:

- `docs/PSION_PROGRAM_MAP.md`
- `docs/PSION_PRETRAIN_STAGE.md`
- `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`

### Tassadar executor lane

Current focus:

- exact-computation executor substrate
- served evidence, capability publication, and benchmark lanes
- article-runbook and bounded public claim surfaces

Start with:

- `docs/ROADMAP_TASSADAR.md`
- `docs/TASSADAR_RUST_ONLY_ARTICLE_RUNBOOK.md`
- `docs/TASSADAR_MULTI_PLUGIN_ORCHESTRATION_WAVE.md`

## Useful Entry Points

### Binaries

- `./TRAIN`
- `cargo run -p psionic-serve --bin psionic-gpt-oss-server -- --help`
- `cargo run -p psionic-train --bin parameter_golf_single_h100_train -- --help`
- `cargo run -p psionic-train --bin psion_google_two_node_configured_peer_open_adapter_swarm -- --help`

### Checks and scripts

- `scripts/benchmark-gpt-oss-vs-llama.sh`
- `scripts/check-parameter-golf-runpod-8xh100-lane.sh`
- `scripts/check-psion-google-two-node-swarm-runbook.sh`
- `scripts/check-first-swarm-trusted-lan.sh`

## Audits

`docs/audits/` explains why the program is moving in a given direction and
records closeout truth. It is useful context, but it is not the canonical
current-state spec.
