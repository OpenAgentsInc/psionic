# CSM Audio Runtime

Status: partial

This document tracks the Psionic-owned CSM speech-generation lane for Lyra.
CSM is a contextual speech generator. It is not the Lyra conversation runtime,
STT engine, LLM, transport, or product authority layer.

The current implementation state is phase 6:

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
- `psionic-models` now owns the first Rust CPU CSM generation path through an
  in-process Rust Candle CSM model: it loads `sesame/csm-1b` safetensors,
  validates model/config digests, accepts Rust-built prompt frames, generates
  32-codebook audio frames, records deterministic frame digests, and can decode
  the generated frames through Rust Mimi into WAV-ready PCM.
- Approved voice profiles publish precomputed prompt-codebook descriptors with
  provenance, sample rate, codebook count, frame counts, and token digests.
- Psionic has a committed voice-profile governance manifest at
  `fixtures/csm/voice_profiles/lyra_voice_profiles.v1.json` and a
  machine-readable schema at
  `fixtures/csm/voice_profiles/lyra_voice_profiles.schema.json`.
- The first governed profile is `lyra/default_female_v1`. It maps to the
  committed source prompt `conversational_a`, is admitted only for local and
  internal Lyra development, and is not a public voice-cloning or production
  consent grant.
- `psionic-serve` exposes a Rust-only CSM speech API surface through
  `psionic-csm-speech-server`.
- That server publishes `/health`, `/v1/models`, `POST /v1/audio/speech`, and
  `POST /psionic/csm/speech`.
- On hosts with the gated artifacts in the local Hugging Face cache, the server
  warm-loads the Rust tokenizer, CSM model, and Mimi decoder at startup and can
  answer repeated short `wav` speech requests.
- `stream=true` returns a buffered `multipart/mixed` response with ordered
  `audio/wav` chunks and terminal JSON metadata. This is not frame-by-frame
  low-latency decode yet; it is the first Lyra-compatible chunked transport.
- Metal/CUDA CSM acceleration is not claimed. The server publishes
  `accelerated_backend = unavailable_fail_closed` while the admitted live
  backend is warm CPU.
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

Useful environment controls:

- `PSIONIC_CSM_RUNTIME=disabled` starts the API in fail-closed publication mode
  without loading gated artifacts.
- `PSIONIC_CSM_BACKEND=cpu` is the only admitted live backend today. Other
  values fail closed with `unsupported_backend`.
- `PSIONIC_CSM_HOST`, `PSIONIC_CSM_PORT`, and `PSIONIC_CSM_MODEL_ID` mirror the
  command-line host, port, and model controls.

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
- `stream`, returning buffered multipart audio chunks when `true`
- `psionic_csm.temperature`
- `psionic_csm.top_k`
- `psionic_csm.max_audio_length_ms`, currently `80..=2000` on warm CPU
- `psionic_csm.context_policy`, currently `none` for served requests

If no voice is provided, the server defaults to `lyra/default_female_v1`.
Requests for raw fixture prompt ids such as `conversational_a` or
`conversational_b` are refused because served requests must use governed Lyra
profile ids.

The route is intentionally not backed by Python. If any gated local artifact is
missing, incompatible, disabled, or requested on an unsupported backend, a
valid speech request returns a structured fail-closed refusal with a specific
code such as:

- `runtime_disabled`
- `llama_tokenizer_unavailable`
- `csm_config_unavailable`
- `csm_model_unavailable`
- `mimi_model_unavailable`
- `unsupported_backend`

Ready responses publish:

- `served_backend = cpu`
- `execution_mode = native`
- `execution_engine = rust_candle_csm_cpu`
- `residency = warm_cpu`

The response also includes execution and artifact headers such as
`x-psionic-model-id`, `x-psionic-execution-engine`,
`x-psionic-csm-voice-profile-id`, CSM artifact digest headers,
`x-psionic-first-audio-latency-ms`,
`x-psionic-full-generation-latency-ms`, `x-psionic-output-duration-ms`,
`x-psionic-csm-frames-sha256`, and `x-psionic-csm-wav-pcm16-digest`.

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
- safety capability truth: watermarking is published as
  `unsupported_fail_closed` with `csm_watermarking_unavailable`, so CSM output
  is blocked for production Lyra cutover until voice governance and watermark
  policy are implemented
- runtime truth: `ready`/`unavailable`, warm-load latency, backend,
  residency, artifact availability, and accelerated-backend refusal truth

One-shot local request:

```bash
curl -D /tmp/psionic-csm-wav.headers \
  -o /tmp/psionic-csm.wav \
  -X POST http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"sesame/csm-1b","input":"hello from psionic","voice_profile_id":"lyra/default_female_v1","response_format":"wav","psionic_csm":{"max_audio_length_ms":160,"context_policy":"none"}}'
```

Buffered multipart stream request:

```bash
curl -D /tmp/psionic-csm-stream.headers \
  -o /tmp/psionic-csm.multipart \
  -X POST http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"sesame/csm-1b","input":"hello from psionic","voice_profile_id":"lyra/default_female_v1","response_format":"wav","stream":true,"psionic_csm":{"max_audio_length_ms":160,"context_policy":"none"}}'
```

## Voice Profile Governance

The served voice-profile contract is now:

- public served id: `lyra/default_female_v1`
- source prompt profile: `conversational_a`
- approval status: `approved_internal_placeholder`
- runtime admission: `admitted_internal_development`
- allowed surfaces: `psionic_local_development` and `lyra_internal_development`
- disallowed surfaces: `lyra_production`, `public_user_voice_clone`, and
  `arbitrary_reference_audio_upload`
- source provenance:
  `committed_csm_parity_fixture_prompt_conversational_a`
- consent posture:
  `internal_placeholder_from_committed_reference_prompt_not_arbitrary_user_upload`

The route refuses unknown or ungoverned profile ids with
`voice_profile_unavailable`. This is intentional: prompt fixtures are not
served voice ids. A source prompt can feed a governed profile, but callers must
use the governed Lyra profile id.

Arbitrary voice cloning and reference-audio upload are out of scope until a
consent system exists. The current Rust Mimi encode capability remains refused
with `rust_mimi_encode_not_implemented`, and CSM watermarking remains
`unsupported_fail_closed` with `csm_watermarking_unavailable`.

Public demo watermark keys are not production safety controls. Production
Lyra cutover remains blocked until Psionic has a private watermark or
equivalent voice-safety control and the governed profile moves beyond internal
placeholder status.

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
- `CsmCpuGenerator::from_safetensors_file(...)`
- `CsmCpuGenerator::generate_codebook_frames(...)`
- `CsmCpuGenerator::generate_and_decode(...)`
- `csm_codebook_frames_digest(...)`

Tokenizer loading is native Rust through the `tokenizers` crate. The served
path does not start Python and does not call the local reference repo. CSM
generation is native Rust through Candle Transformers inside `psionic-models`;
the code does not shell to Python or use a Python worker. When the matching
gated Llama tokenizer JSON, CSM config, CSM model weights, and Mimi weights are
present in the local Hugging Face cache, the focused test runs text-only Rust
CSM generation and Rust Mimi decode end to end.

## Rust CPU Generation

The first CSM generation engine is:

- backend: `cpu`
- execution engine: `rust_candle_csm_cpu`
- model: `sesame/csm-1b`
- sampler: greedy argmax or seeded top-k
- input: `CsmPromptFramePlan`
- output: 32-lane codebook frames plus stable SHA-256 digest
- decode path: `rust_moshi_mimi_cpu`

The path is intentionally bounded. It proves that Psionic can load the gated
CSM weights and produce audio frames entirely in Rust. The HTTP server now
keeps that CPU path warm and serves short WAV responses plus buffered multipart
chunks. It is still correctness-first rather than production-latency-ready.
Metal/CUDA acceleration, true frame-by-frame audio streaming, prompt-codebook
context use, production voice consent, and production watermark posture remain
separate cutover gates.

Exact replay of the committed Python generation fixture remains unavailable
because the fixture stores only compact prompt-codebook prefixes, not the full
prompt codebook tensor needed to reconstruct the original prompted context.
Psionic publishes this as an explicit fixture gap rather than pretending parity
coverage exists. Exact prompted replay needs either full committed prompt
codebooks or a Rust Mimi encode path.

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
- CSM config digest:
  `sha256:b203c014cb5a2f7b4f98d2e945f091182aceb17fa530ce968e8c3437e01a9b70`
- CSM model digest:
  `sha256:2e7721144afe38b906d4f1048671da639fe142423f4a26283606ecebe894f4bf`
- Llama tokenizer digest:
  `sha256:79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`
- prompt profiles: `conversational_a`, `conversational_b`
- source prompt WAVs: 44.1 kHz mono, 30 seconds each
- CSM runtime sample rate: 24 kHz
- CSM text/audio frame width: 33 lanes
- CSM audio codebook count: 32
- deterministic generation prefix sampling: `greedy_argmax_topk1`

## Retained Serving Smoke

The retained local smoke report is:

- `fixtures/csm/reports/csm_warm_cpu_serving_smoke_2026-05-06.json`
- `fixtures/csm/reports/csm_voice_governance_smoke_2026-05-06.json`

It records a warm CPU server on `127.0.0.1:18083` with:

- health `status = ok`
- `runtime.residency = warm_cpu`
- one-shot `POST /v1/audio/speech` returning `200` and `audio/wav`
- playable WAV output: RIFF/WAVE, PCM16 mono, 24 kHz
- buffered multipart stream returning `200` and
  `multipart/mixed; boundary=psionic-csm-stream`
- output duration: `160 ms`
- generated CSM frame count: `2`
- one-shot full-generation latency: `2702 ms`
- stream full-generation latency: `2677 ms`

The voice-governance smoke records:

- `/health` publishing `lyra/default_female_v1`
- source prompt mapping: `conversational_a`
- raw `conversational_a` speech requests refused with
  `voice_profile_unavailable`
- governed `lyra/default_female_v1` speech request returning `200 audio/wav`
- headers publishing approval status and fail-closed watermarking posture

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
- Rust CPU CSM generation and Rust Mimi decode when the matching local CSM,
  Llama-tokenizer, and Mimi artifacts are available
- warm CSM speech serving when the matching local artifacts are available
- buffered multipart stream framing with ordered binary WAV chunks and terminal
  metadata
- governed Lyra voice-profile admission and raw fixture prompt refusal
- explicit fixture-gap truth for exact deterministic prompted replay
- explicit refusal truth for runtime reference-audio encoding

## Next Phases

The phase sequence lives in GitHub under `OpenAgentsInc/psionic#959`.

Next work:

1. Integrate Lyra through the Psionic TTS provider boundary after generation
   returns real audio bytes.
2. Add Metal/CUDA acceleration and true frame-by-frame low-latency decode when
   CSM quality and voice governance justify product cutover work.

Cartesia remains Lyra's production TTS provider until CSM has measured quality,
latency, approved voice-profile governance, and watermark posture.
