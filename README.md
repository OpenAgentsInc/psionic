# Psionic

Psionic is a Rust-native ML and inference stack.

It owns the machine-facing execution substrate behind local inference, serving,
training, distributed execution, artifact truth, and clustered compute. The
project is broader than one app or one benchmark lane. It is the crate family
that OpenAgents uses for inference, training, cluster bring-up, and execution
evidence.

Psionic should be read hardware-first. It owns the admitted hardware strategy
for each lane: backend family, residency mode, topology, serving or training
role, and the capability, refusal, and evidence surfaces that higher layers
consume. Upstream systems such as `llama.cpp`, `vLLM`, `SGLang`, MLX, and
other reference repos are inputs for specific layers or hardware classes, not
the identity of the shipped Psionic stack.

The training side now also carries one bounded `gemma4:e4b` CUDA adapter-SFT
trainer above the shared adapter substrate: LM-head-only final-hidden-state
supervision, frozen-base semantics, typed export, exact checkpoint resume,
served-base plus tokenizer compatibility checks, and explicit refusal truth for
wider Gemma regions that remain out of scope. The same bounded lane now also
closes the first trainer-to-serving refresh seam: typed Gemma checkpoints plus
exported adapter artifacts can be revalidated into the live CUDA mesh lane
without a process restart, the active served revision is surfaced in response
provenance, stale or mismatched revisions fail closed, and operators can roll
back to the last known-good promoted revision. The same lane is now also
eval-first: it binds one canonical held-out eval pack, one four-split dataset
contract, one short baseline sweep against the untuned base, one overlap and
decontam gate, one canned promoted-checkpoint vibe-review packet, and one
promotion decision that refuses held-out regressions or failed operator review.

## Start Here

- System architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Detailed workspace map: [docs/WORKSPACE_MAP.md](docs/WORKSPACE_MAP.md)
- Inference and serving: [docs/INFERENCE_ENGINE.md](docs/INFERENCE_ENGINE.md)
- Inference mesh ownership: [docs/INFERENCE_MESH_OWNERSHIP.md](docs/INFERENCE_MESH_OWNERSHIP.md)
- Mesh lane service mode: [docs/MESH_LANE_SERVICE_MODE.md](docs/MESH_LANE_SERVICE_MODE.md)
- Optimizer substrate: [docs/OPTIMIZER_SUBSTRATE.md](docs/OPTIMIZER_SUBSTRATE.md)
- Forge-facing eval pack publication: [docs/PSION_FORGE_EVAL_PACK_MANIFESTS.md](docs/PSION_FORGE_EVAL_PACK_MANIFESTS.md)
- Hermes user guide: [docs/hermes/README.md](docs/hermes/README.md)
- Training system: [docs/TRAIN_SYSTEM.md](docs/TRAIN_SYSTEM.md)
- Repo-local library roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Psion learned-model program: [docs/PSION_PROGRAM_MAP.md](docs/PSION_PROGRAM_MAP.md)
- Psion actual-pretraining operator runbook: [docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md](docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md)
- Psion bounded reference-lane smoke runbook: [docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md](docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md)

## Main Tracks

- Inference and local serving
  - local GPT-OSS server and benchmark harness
  - generic OpenAI-compatible server surfaces
  - hardware validation and backend truth
  - bounded non-`GptOss` lanes including `qwen35`, the published dense
    `gemma4:e4b` CUDA lane, the sparse `gemma4:26b` topology-publication and
    refusal lane, and the optional dense `Gemma 4 31B` validation repeat that
    keeps the same family contract without widening the first claim
  - the Gemma image or video path now publishes as a processor-owned refusal
    lane instead of pretending the dense text surface can consume media URLs
  - the dense `Gemma 4` `e2b` and `e4b` rows now also publish a separate
    processor-owned audio lane with explicit `input_audio` refusal until the
    real audio processor lands, while `31B` and `26B` still fail closed
  - the generic server now also publishes one first-class `Gemma 4` Metal lane
    contract with `backend = metal`, `execution_mode = native`, and
    `fallback_policy = refuse`, and it returns an explicit refusal instead of
    silently falling back to CPU or CUDA until a real Metal decoder lands
  - the generic server, routed inventory, and mesh management status now also
    publish family-agnostic clustered execution truth so downstream consumers
    can tell whether a model is remote-proxied, replicated, split across
    machines, or running as a sparse distributed expert row without `gpt_oss`
    specific heuristics
  - start with [docs/GPT_OSS_LOCAL_SERVING.md](docs/GPT_OSS_LOCAL_SERVING.md)
  - supporting docs: [docs/NON_GPT_OSS_QWEN35_PILOT.md](docs/NON_GPT_OSS_QWEN35_PILOT.md), [docs/NON_GPT_OSS_GEMMA4_PILOT.md](docs/NON_GPT_OSS_GEMMA4_PILOT.md)
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
  - optional mesh coordination adjunct under `/psionic/management/coordination/*` for typed
    status, finding, question, tip, and done packets with TTL, visibility, provenance, search,
    and redaction semantics outside the inference critical path
  - expert-family GGUF admission now stays explicit: `psionic-models` can inspect non-`gpt-oss`
    expert artifacts, carry artifact identity plus expert-topology requirements, and refuse native
    execution with a machine-checkable topology-contract error instead of collapsing them into a
    generic unsupported-family bucket
  - `psionic-cluster` now also owns one native sparse expert-placement contract over explicit
    expert-host inventory, stable placement digests, typed refusal codes, and reusable sharded
    execution receipts instead of a sidecar-only MoE control plane; the first specialized lane is
    `gemma4:26b` with `64` experts, `4` active experts, `family_specific_placement`, and a
    truthful two-host partitioned planning policy
  - start with [docs/ROADMAP_CLUSTER.md](docs/ROADMAP_CLUSTER.md)
  - supporting docs: [docs/INFERENCE_MESH_OWNERSHIP.md](docs/INFERENCE_MESH_OWNERSHIP.md), [docs/MESH_LANE_SERVICE_MODE.md](docs/MESH_LANE_SERVICE_MODE.md), [docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md](docs/FIRST_SWARM_TRUSTED_LAN_RUNBOOK.md), [docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md](docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md), [docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md](docs/TRAIN_ARTIFACT_STORAGE_REFERENCE.md)
- Psion learned-model program
  - corpus, tokenizer, pretrain, trusted-cluster, and decentralized contribution work
  - start with [docs/PSION_PROGRAM_MAP.md](docs/PSION_PROGRAM_MAP.md)
  - supporting docs: [docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md](docs/PSION_ACTUAL_PRETRAINING_RUNBOOK.md), [docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md](docs/PSION_LOCAL_FIRST_TRAIN_RUNBOOK.md), [docs/PSION_PRETRAIN_STAGE.md](docs/PSION_PRETRAIN_STAGE.md), [docs/PSION_TRUSTED_CLUSTER_RUN.md](docs/PSION_TRUSTED_CLUSTER_RUN.md), [docs/PSION_DECENTRALIZED_CONTRIBUTION.md](docs/PSION_DECENTRALIZED_CONTRIBUTION.md)

## Psion Training Shortcut

If you want the current top Psion training lane instead of guessing among
benchmark-adjacent lanes, run:

```bash
./TRAIN
```

That command now targets the actual Psion pretraining lane and materializes the
retained launch, status, preflight, checkpoint, dashboard, alert, and closeout
surfaces under `~/scratch/psion_actual_pretraining_runs/<run_id>`.

Use:

```bash
./TRAIN --dry-run
./TRAIN resume --run-root <path>
./TRAIN status --run-root <path>
```

for plan inspection and operator follow-up on the actual lane.

For machine supervision, use the typed runtime surface instead of the shell
wrapper:

```bash
cargo run -q -p psionic-train -- manifest --manifest <path-to-psionic.train.invocation_manifest.v1.json>
```

That manifest now carries one frozen coordination envelope, including the
admitted `node_pubkey`, plus one admitted release/build/environment identity
that the runtime checks before launch. Recovery-source manifests can now also
carry one `peer_node_pubkey` for the machine-only `serve-checkpoint` operation
and one `peer_checkpoint_handoff_receipt_path` that seeds a joiner’s local
checkpoint tree before `resume` runs. Validator manifests are now admitted for
one machine-only `validate-contribution` operation as well, using one
`validator_target_contribution_receipt_path` plus one
`validator_target_contribution_artifact_manifest_path` to replay one retained
worker contribution artifact set into one local validator score receipt. The emitted
`psionic.train.status_packet.v1` packet now also carries the resolved runtime
attestation and the retained absolute paths for
`status/psionic_train_run_status_packet.json` and
`status/psionic_train_window_status_packet.json`. When a run root exists, the
runtime also persists one `psionic.train.membership_revision_receipt.v1`
receipt at `status/membership_revision_receipt.json` and appends revision
history under `status/membership_revisions/`. That same machine contract now
admits one second bounded lane, `psion_apple_windowed_training_v1`, for
homogeneous Apple Silicon / Metal windowed training. The Apple lane uses the
same invocation manifest, status packets, membership receipt, contribution
artifacts, validator replay entrypoint, and peer handoff flow, but it retains
generic checkpoint artifacts under
`checkpoints/latest_accepted_checkpoint_pointer.json` plus
`checkpoints/manifests/checkpoint_manifest_step-<optimizer_step>.json` using
`psionic.train.checkpoint_pointer.v1` and
`psionic.train.checkpoint_manifest.v1`. That is intentionally narrower than the
actual pretraining lane: it is one admitted machine lane for backend-homogeneous
Apple windows, not a claim that the broader CUDA actual-pretraining operator
contract is now portable across backend families.
history under `status/membership_revisions/` so the local worker heartbeat,
drain, rejoin, replace, and failed-session posture remain machine-visible.
The same machine runtime now also persists one
`psionic.train.checkpoint_surface.v1` snapshot at
`status/checkpoint_surface.json` so supervisors can read the latest checkpoint
pointer state, checkpoint-manifest digest, backup receipt posture, upload
outcome, and auto-resume recovery result without reopening the full retained
actual-lane tree. The run/window status packets repeat the absolute paths for
that surface plus the latest checkpoint manifest, backup receipt, pointer,
peer handoff receipt, auto-resume receipt, and validator score receipt when
those artifacts exist. Validator replay retains the score surfaces under
`windows/<window_id>/validators/<challenge_id>/validator_score_artifact.json`
and `validator_score_receipt.json`.
The machine validator contract is now also covered by focused unit tests over
disposition classification and subprocess CLI tests over stale assignment,
missing replay inputs, and artifact-digest drift refusals.
When the admitted coordination envelope also carries `window_id` and
`assignment_id`, the same machine runtime now materializes one deterministic
window artifact family under `windows/<window_id>/`: one retained
`window_execution.json`, one per-contribution `artifact_manifest.json`, one
per-contribution `contribution_receipt.json`, and one rollup
`sealed_window_bundle.json`. The run/window status packets repeat the absolute
paths for those retained window surfaces too, so supervisors can follow one
declared assignment through the local retained bundle set without re-scanning
the whole run root.

The older bounded reference pilot still exists as the smoke/reference lane:

```bash
./TRAIN --lane reference_pilot --dry-run
./TRAIN --lane reference_pilot --mode local_reference
```

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

The bounded default-lane rehearsal lives in
[docs/TASSADAR_DEFAULT_TRAIN_REHEARSAL.md](docs/TASSADAR_DEFAULT_TRAIN_REHEARSAL.md).

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

## Installable Mesh Lanes

Psionic also ships `crates/psionic-serve/src/bin/psionic-mesh-lane.rs` as the
supported service-mode entrypoint for durable inference-mesh nodes.

It materializes one lane root with config, file-backed node identity, durable
network state, logs, model paths, and generated `launchd` / `systemd` service
artifacts. `openagents` and `probe` integrate against that Psionic-owned
service binary and its management surfaces directly; the supported pooled
inference path does not depend on any separate mesh sidecar runtime. The full
operator runbook lives in
[docs/MESH_LANE_SERVICE_MODE.md](docs/MESH_LANE_SERVICE_MODE.md).

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
