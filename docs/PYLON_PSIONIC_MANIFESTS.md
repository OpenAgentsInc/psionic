# Pylon Psionic Manifests

Status: `implemented_early`

Psionic publishes Pylon-consumable manifests for optional local Qwen3.5
inference. Pylon does not bundle Psionic binaries or Qwen weights by default.
An operator must explicitly choose the install path, pass machine checks, and
verify SHA-256 digests before Pylon places any artifact in its cache.

## Release Manifest

Schema consumed by Pylon:

- `openagents.psionic.release_manifest.v0.3`

Contract marker:

- `psionic.release_manifest.v1`

Fixture manifests:

- `fixtures/pylon/psionic/release_manifest_darwin_arm64_v0_3.json`
- `fixtures/pylon/psionic/release_manifest_linux_x64_v0_3.json`
- `fixtures/pylon/psionic/release_manifest_linux_arm64_v0_3.json`

Each release manifest includes:

- selected platform row for Pylon's current installer;
- first-pass platform rows for `darwin-arm64`, `linux-x64`, and
  `linux-arm64`;
- `psionic-openai-server` binary ref, artifact ref, SHA-256 digest, and
  signature ref;
- supported endpoint truth for `/health`, `/v1/models`, and
  `/v1/chat/completions`;
- backend family truth for `cpu`, `cuda`, and `metal`;
- explicit `inferenceOnly = true`;
- explicit blocked `trainingClaim` and `paidInferenceClaim`.

## Model Artifact Manifests

Schema consumed by Pylon:

- `openagents.psionic.model_artifact_manifest.v0.3`

Contract marker:

- `psionic.model_artifact_manifest.v1`

Fixture manifests:

- `fixtures/pylon/psionic/model_artifact_manifest_qwen35_0_8b_q8_0_v0_3.json`
- `fixtures/pylon/psionic/model_artifact_manifest_qwen35_2b_q8_0_v0_3.json`

The 0.8B row is a low-footprint smoke and fallback model:

- model key: `qwen35-0_8b-q8_0`
- model ref: `model.psionic.qwen35.0_8b.q8_0`
- role: `low_footprint_smoke_fallback`

The 2B row is the first coding-agent and tool-loop quality row:

- model key: `qwen35-2b-q8_0`
- model ref: `model.psionic.qwen35.2b.q8_0`
- role: `coding_agent_tool_loop`

Both rows carry artifact refs, SHA-256 digests, model family, parameter class,
quantization, chat-template digest, license boundary, admitted backend
families, admitted endpoints, and tool-calling smoke refs.

## Pylon Behavior

Pylon should:

- reject unsupported platforms before fetching;
- require operator consent before fetching;
- verify the release manifest and selected platform;
- verify binary bytes against the manifest SHA-256 before placement;
- verify model bytes against the model manifest SHA-256 before placement;
- publish only backend/model/digest/cache refs in public closeouts;
- attach to a running `psionic-openai-server` through `/health`,
  `/v1/models`, and `/v1/chat/completions`.

Pylon should not:

- claim Qwen training is live from these manifests;
- claim paid Qwen inference is live from these manifests;
- imply the artifacts are bundled with `@openagentsinc/pylon`;
- expose local model paths, prompt text, authorization material, or private
  machine topology in public projections.

## Validation

The validation implementation lives in:

- `crates/psionic-serve/src/pylon_release_manifest.rs`

Run:

```sh
cargo test -p psionic-serve pylon_manifest
```

The tests validate all fixture manifests, require all first-pass platform rows,
require the OpenAI-compatible endpoint set, require tool-call admission on both
model rows, and reject private-path or training-overclaim rows.
