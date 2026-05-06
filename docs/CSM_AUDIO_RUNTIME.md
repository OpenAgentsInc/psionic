# CSM Audio Runtime

Status: partial

This document tracks the Psionic-owned CSM speech-generation lane for Lyra.
CSM is a contextual speech generator. It is not the Lyra conversation runtime,
STT engine, LLM, transport, or product authority layer.

The current implementation state is phase 1:

- Psionic has a committed Python-reference parity corpus at
  `fixtures/csm/python_reference/csm_python_parity_v1.json`.
- `psionic-models` validates that fixture through
  `csm_python_parity_fixture()` and `validate_csm_python_parity_fixture(...)`.
- The fixture freezes prompt audio hashes, Llama tokenizer examples, 33-lane
  CSM text-frame masks, compact Mimi prompt-codebook prefixes, and a
  three-frame greedy generated-codebook prefix.
- `psionic-serve` exposes a Rust-only CSM speech API surface through
  `psionic-csm-speech-server`.
- That server publishes `/health`, `/v1/models`, `POST /v1/audio/speech`, and
  `POST /psionic/csm/speech`.
- The speech request route currently validates request shape and then refuses
  with `rust_csm_generation_not_implemented` until the Rust tokenizer, Mimi,
  safetensors, and generation phases land.
- The local Python repo at `/Users/christopherdavid/code/csm` remains a
  reference harness and parity source only. It is not a production Psionic
  runtime, it is not embedded in Lyra, and it is not called by the Psionic
  service path.

## Fixture Source

The frozen corpus was derived from the local CSM demo described in:

- `/Users/christopherdavid/code/csm/DEMO_RUN.md`
- root audit:
  `/Users/christopherdavid/work/docs/2026-05-06-csm-rust-lyra-psionic-audit.md`

The reference demo command is:

```bash
NO_TORCH_COMPILE=1 .venv/bin/python run_csm.py
```

The fixture records no Hugging Face token, provider key, full prompt audio,
full model weights, or full codebook tensors.

## Rust-Only Service Surface

Start the current Rust CSM speech server with:

```bash
cargo run -p psionic-serve --bin psionic-csm-speech-server -- --host 127.0.0.1 --port 8081
```

The current endpoints are:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `POST /psionic/csm/speech`

The request shape accepts:

- `model`, defaulting to `sesame/csm-1b`
- `input`
- `voice` or `voice_profile_id`
- `response_format`, currently only `wav`
- `stream`, currently refused
- `psionic_csm.temperature`
- `psionic_csm.top_k`
- `psionic_csm.max_audio_length_ms`
- `psionic_csm.context_policy`

The route is intentionally not backed by Python. Until the Rust model path is
implemented, a valid speech request returns a structured `503` refusal with:

- `code = rust_csm_generation_not_implemented`
- `served_backend = cpu`
- `execution_mode = native`
- `execution_engine = rust_csm_pending`

The response also includes execution and artifact headers such as
`x-psionic-model-id`, `x-psionic-execution-engine`,
`x-psionic-csm-voice-profile-id`, and CSM artifact digest headers.

## Current Fixture Contents

The fixture binds:

- CSM repo: `sesame/csm-1b`
- Llama tokenizer repo: `meta-llama/Llama-3.2-1B`
- Mimi repo: `kyutai/moshiko-pytorch-bf16`
- Mimi weight: `tokenizer-e351c8d8-checkpoint125.safetensors`
- prompt profiles: `conversational_a`, `conversational_b`
- source prompt WAVs: 44.1 kHz mono, 30 seconds each
- CSM runtime sample rate: 24 kHz
- CSM text/audio frame width: 33 lanes
- CSM audio codebook count: 32
- deterministic generation prefix sampling: `greedy_argmax_topk1`

## Validation

Run the focused fixture validation with:

```bash
cargo test -p psionic-models csm_python_parity_fixture
```

Run the served API/refusal tests with:

```bash
cargo test -p psionic-serve csm_
```

The validator checks:

- fixture schema and artifact digest shapes
- required prompt profile ids
- prompt WAV metadata
- tokenizer frame dimensions and text-lane mask semantics
- Mimi codebook prefix dimensions and token bounds
- deterministic generation frame dimensions and token bounds
- explicit secret-redaction markers

## Next Phases

The phase sequence lives in GitHub under `OpenAgentsInc/psionic#959`.

Next work:

1. Implement Rust tokenizer, prompt framing, and artifact descriptors.
2. Implement Mimi decode and approved voice-profile codebook support.
3. Implement CPU CSM generation with parity tests.
4. Add accelerated serving, residency/refusal truth, and streaming chunks.

Cartesia remains Lyra's production TTS provider until CSM has measured quality,
latency, approved voice-profile governance, and watermark posture.
