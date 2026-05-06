# CSM Audio Runtime

Status: partial

This document tracks the Psionic-owned CSM speech-generation lane for Lyra.
CSM is a contextual speech generator. It is not the Lyra conversation runtime,
STT engine, LLM, transport, or product authority layer.

The current implementation state is phase 3:

- Psionic has a committed Python-reference parity corpus at
  `fixtures/csm/python_reference/csm_python_parity_v1.json`.
- `psionic-models` validates that fixture through
  `csm_python_parity_fixture()` and `validate_csm_python_parity_fixture(...)`.
- The fixture freezes prompt audio hashes, Llama tokenizer examples, 33-lane
  CSM text-frame masks, compact Mimi prompt-codebook prefixes, and a
  three-frame greedy generated-codebook prefix.
- `psionic-models` now owns the Rust CSM frontend contract:
  tokenizer loading from cached Hugging Face `tokenizer.json`, speaker-tag
  text encoding, BOS/EOS template installation, 33-lane text/audio frame
  construction, prompt-window validation, segment-boundary context truncation,
  CSM `config.json` parsing, and artifact/voice-profile descriptors.
- `psionic-models` now owns the first Rust Mimi decode path through the Rust
  `moshi` crate: it loads Kyutai Mimi safetensors, validates weight digests,
  accepts 32-codebook RVQ frames, strips trailing all-zero EOS frames, decodes
  to 24 kHz mono samples, and writes browser-playable PCM16 WAV bytes.
- Approved voice profiles publish precomputed prompt-codebook descriptors with
  provenance, sample rate, codebook count, frame counts, and token digests.
- `psionic-serve` exposes a Rust-only CSM speech API surface through
  `psionic-csm-speech-server`.
- That server publishes `/health`, `/v1/models`, `POST /v1/audio/speech`, and
  `POST /psionic/csm/speech`.
- The speech request route currently validates request shape and then refuses
  with `rust_csm_generation_not_implemented` until the Rust CSM model
  safetensors binding and generation loop land.
- The local Python repo at `/Users/christopherdavid/code/csm` remains a
  reference harness and parity source only. It is not a production Psionic
  runtime, it is not embedded in Lyra, and it is not called by the Psionic
  service path.
- There is no Python worker in this path. Psionic does not shell to Python,
  proxy to the local CSM repo, embed Python, or depend on the Python Moshi
  package at runtime.

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

The `/health` and `/v1/models` surfaces now also publish a Rust-built artifact
descriptor containing:

- CSM, Llama-tokenizer, and Mimi repo ids
- Mimi weight filename
- config, model, tokenizer, and Mimi weight digests
- admitted prompt voice profiles
- admitted prompt-codebook descriptor digests for approved profiles
- frame contract: 33 lanes, 32 audio lanes, text lane 32, max sequence length
  2048, 80 ms generation frames, 24 kHz runtime audio
- codec capability truth: Mimi decode implemented by `rust_moshi_mimi_cpu`,
  runtime reference-audio encoding refused with
  `rust_mimi_encode_not_implemented`

## Rust Frontend Contract

The Rust model frontend lives in `crates/psionic-models/src/csm.rs`.

It provides:

- `CsmLlamaTextTokenizer::from_tokenizer_json_file(...)`
- `CsmLlamaTextTokenizer::from_default_hf_cache(...)`
- `csm_format_segment_text(speaker, text)` using `[{speaker}]{text}`
- `csm_text_frame_block(...)`
- `csm_audio_frame_block(...)`, including the all-zero codebook EOS frame
- `CsmPromptSegment`
- `csm_build_prompt_frame_plan(...)`
- `CsmModelConfig::from_json_str(...)`
- `CsmModelArtifactDescriptor::from_fixture(...)`

Tokenizer loading is native Rust through the `tokenizers` crate. The served
path does not start Python and does not call the local reference repo. When the
matching gated Llama tokenizer JSON is present in the local Hugging Face cache,
the focused test compares Rust token IDs with the frozen Python fixture.

## Rust Mimi Decode

The Rust Mimi decoder lives in `crates/psionic-models/src/csm.rs`.

It provides:

- `CsmMimiDecoder::from_safetensors_file(...)`
- `CsmMimiDecoder::decode_codebook_frames(...)`
- `csm_generation_case_codebook_frames(...)`
- `CsmAudioClip::to_wav_pcm16()`
- `csm_wav_pcm16_digest(...)`
- `csm_reference_audio_encoding_refusal()`

The first decode implementation uses the Rust `moshi` crate in-process on CPU.
That is allowed because it is Rust Psionic code, not the Python CSM repo and
not the Python Moshi package. The local deterministic fixture currently decodes
to:

- clip digest:
  `sha256:30350d2c6648102458e2eedb3c2388894b162452de6fbce931f1058f95d9c509`
- PCM16 WAV digest:
  `sha256:8a23a6965b90c0faf627f3eb203c45c8fafc4200c7d8e96231660c4cd931e0cd`

Runtime reference-audio encoding is intentionally unsupported until a Rust
encode path lands. Requests or flows that require encoding arbitrary uploaded
reference audio must fail closed with
`rust_mimi_encode_not_implemented`. The admitted shortcut for now is an
approved profile whose prompt codebooks were precomputed offline and committed
as descriptor digests.

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

Run the Rust frontend/tokenizer/framing tests with:

```bash
cargo test -p psionic-models csm_
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

The frontend tests additionally check:

- speaker-tag formatting, including empty text
- CSM config parsing
- text-frame construction against the frozen fixture
- audio-frame EOS construction
- multi-segment prompt assembly
- max-context refusal
- segment-boundary context truncation
- real tokenizer parity when the matching local HF tokenizer JSON is available
- Mimi codebook decode when the matching local Mimi safetensors file is
  available
- deterministic PCM/WAV digest stability for the local decoded fixture
- explicit refusal truth for runtime reference-audio encoding

## Next Phases

The phase sequence lives in GitHub under `OpenAgentsInc/psionic#959`.

Next work:

1. Implement CPU CSM generation with parity tests.
2. Add accelerated serving, residency/refusal truth, and streaming chunks.
3. Integrate Lyra through the Psionic TTS provider boundary after generation
   returns real audio bytes.

Cartesia remains Lyra's production TTS provider until CSM has measured quality,
latency, approved voice-profile governance, and watermark posture.
