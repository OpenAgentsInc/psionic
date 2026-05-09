# Psionic VAD

Status: implemented_early

Psionic owns the model-backed VAD path for Autopilot Voice. Autopilot should
consume VAD through a Psionic worker API instead of embedding Silero, Candle,
ONNX, browser VAD, or a product-local neural runtime.

## Current MVP

The current crate is `psionic-vad`.

It implements:

- `psionic.vad.v1` worker protocol types;
- per-session start, reset, chunk inference, and flush behavior;
- 48 kHz / 24 kHz / 16 kHz / 8 kHz input admission;
- owned downmix and resampling to 16 kHz inference frames;
- Silero-style 512-sample frame cadence, 64-sample context, threshold,
  negative threshold, minimum speech, minimum silence, speech padding, and
  maximum speech controls;
- recurrent per-session state, candidate speech starts, speech ends, no-speech
  finalization, and stable state digests;
- deterministic fixture smoke coverage.

The MVP is a Psionic-owned Silero-style signal path. It does not call Python,
does not call ONNX Runtime, does not use Candle as the product runtime, and
does not expose Silero internals to Autopilot. Future issues will replace or
back this MVP with a stronger model artifact while preserving the same worker
contract.

## Artifact Manifest

The current retained artifact manifest is:

- `fixtures/vad/model_artifacts/silero_style_vad_mvp_manifest.v1.json`

It pins:

- artifact id: `psionic-vad/silero-style-mvp-v0`
- execution engine: `psionic_silero_style_vad_mvp`
- schema: `psionic.vad.artifact.v1`
- 16 kHz inference, 512-sample frames, and 64-sample context
- parameter digest:
  `sha256:aa67063258e45a67a7c82b92ddace04c4bee9579a22fca4f53ef4aa206d2977f`
- provider keys: not required
- raw private audio retention: not required
- Silero MIT attribution for reference material

Startup validation must reject mismatched artifact ids, execution engines,
sample rates, frame sizes, context sizes, parameter digests, provider-key
requirements, and raw-audio-retention requirements.

## Ownership Boundary

```text
Autopilot
  browser capture, WebSocket session state, voice frames, endpoint decisions,
  HUD handoff, RMS/peak fallback, product telemetry

Psionic
  VAD model/runtime, session inference state, artifact identity, corpus
  benchmarks, promotion gates, worker deployment, latency/resource telemetry

Blueprint
  Program Run, evidence, receipt, policy, and action linkage after a final
  transcript or governed command decision exists
```

## Worker Contract

The library exposes:

- `VadWorkerConfig`
- `VadSessionConfig`
- `VadChunkRequest`
- `VadChunkResponse`
- `VadEndpointEvent`
- `PsionicVadWorker`

Every response carries:

- protocol version;
- session id and chunk index;
- execution engine;
- model artifact id;
- input and inference sample counts;
- processed frame count;
- latest and smoothed speech probabilities;
- speech active state;
- optional endpoint event;
- buffered sample count;
- state digest.

## Local Smoke

Run:

```bash
cargo test -p psionic-vad
cargo run -p psionic-vad --example psionic_vad_fixture_smoke
```

The smoke uses a synthetic fixture and prints machine-readable response frames.
It does not require private audio or provider credentials.

## Next Steps

The next implementation issues should add:

1. model artifact manifest, hash, and license attribution;
2. corpus and replay benchmarks;
3. local service surface with health/readiness;
4. Autopilot `psionic_silero_shadow` integration;
5. promotion gates before any `psionic_silero_primary` endpointing mode.
