# CSM Audio Runtime

Status: Psionic-owned CSM speech worker for Autopilot shadow/canary use

This document tracks the Psionic-owned CSM speech-generation lane for
Autopilot. CSM is a contextual speech generator. It is not the Autopilot
conversation runtime, STT engine, LLM, transport, or product authority layer.
The production runtime decision is recorded in
[CSM Rust Runtime Decision Record](CSM_RUST_RUNTIME_DECISION_RECORD.md).
Artifact, license, voice-profile, and watermark governance is recorded in
[CSM Artifact Governance](CSM_ARTIFACT_GOVERNANCE.md).

The current implementation state is phase 8:

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
- `psionic-models` now owns the Rust CSM generation path through an in-process
  Rust Candle CSM model: it loads `sesame/csm-1b` safetensors, validates
  model/config digests, accepts Rust-built prompt frames, generates 32-codebook
  audio frames, records deterministic frame digests, and can decode the
  generated frames through Rust Mimi into WAV-ready PCM. CPU is the portable
  fallback; CUDA is the admitted accelerator backend for the owned Psionic CSM
  GPU worker.
- Approved voice profiles publish precomputed prompt-codebook descriptors with
  provenance, sample rate, codebook count, frame counts, and token digests.
- Psionic has a committed voice-profile governance manifest at
  `fixtures/csm/voice_profiles/lyra_voice_profiles.v1.json` and a
  machine-readable schema at
  `fixtures/csm/voice_profiles/lyra_voice_profiles.schema.json`.
- The first governed profile is `openagents/default_female_v1`. It maps to the
  committed source prompt `conversational_b`, is admitted for
  OpenAgents-operated Autopilot production dogfood, and is not a public
  voice-cloning or arbitrary reference-audio consent grant.
- `psionic-serve` exposes a Rust-only CSM speech API surface, and the focused
  `psionic-csm-speech` crate builds the production Cloud Run binary without
  compiling unrelated serving surfaces or platform-specific Metal code.
- That server publishes `/health`, `/v1/models`, `POST /v1/audio/speech`, and
  `POST /psionic/csm/speech`.
- The server now also publishes `GET /psionic/csm/worker/metadata` and accepts
  worker fields for `request_id`, `artifact_id`, `timeout_ms`, and
  `cancellation_id` so Autopilot can use one stable Rust RPC boundary for
  shadow/canary evaluation.
- The worker metadata, health response, model card, response headers, and
  stream terminal metadata now publish artifact governance fields including
  artifact id/hash, license posture, runtime image ref, quantization posture,
  governed voice profile ids, watermark status, promotion gate, and rollback
  target.
- On hosts with the gated artifacts in the local Hugging Face cache, the server
  warm-loads the Rust tokenizer, CSM model, and Mimi decoder at startup and can
  answer repeated short `wav` speech requests.
- `stream=true` now returns a generation-time windowed response instead of
  waiting for a full WAV before building the body. The default stream format is
  `multipart_mixed`; Autopilot may request `stream_format=jsonl_base64` to
  receive newline-delimited audio events that carry base64 WAV chunks and
  terminal JSON metadata. This is the first low-latency serving primitive; it
  still uses bounded CSM/Mimi decode windows rather than final production
  chunking, cancellation, and canary evidence.
- CUDA CSM acceleration is admitted behind explicit backend truth. A CUDA
  worker must publish `served_backend = cuda`, `runtime.backend = cuda`,
  `execution_engine = rust_candle_csm_cuda`, and
  `accelerated_backend = cuda`. CPU fallback is allowed only when
  `PSIONIC_CSM_CPU_FALLBACK_ON_ACCELERATOR_FAILURE=true` is deliberately set
  and must publish a CPU fallback reason.
- The local Python repo at `/Users/christopherdavid/code/csm` remains a
  reference harness and parity source only. It is not a production Psionic
  runtime, it is not embedded in Autopilot, and it is not called by the
  Psionic service path.
- There is no Python worker in this path. Psionic does not shell to Python,
  proxy to the local CSM repo, embed Python, or depend on the Python Moshi
  package at runtime.

## Current Production GPU Worker

The current Autopilot production dogfood worker is:

```text
service: psionic-csm-gpu-1
platform: GCE
zone: us-east4-a
accelerator: NVIDIA L4
static IP: 34.48.128.199
endpoint: http://34.48.128.199:8081/v1/audio/speech
image: us-central1-docker.pkg.dev/openagents-lyra/lyra/psionic-csm-speech:211f4ca0
```

This is the active owned GPU route because Cloud Run GPU allocation quota is
not yet available for the `psionic-csm-speech` service. Cloud Run remains the
preferred managed target once quota is granted, but production must not claim
Cloud Run GPU residency until `/health` and speech-response headers prove
CUDA execution on that platform.

The GCE container must prefer host-driver libraries and avoid forcing the
container's `cuda-compat-12-4` driver shim ahead of the host driver. The active
GCE runtime uses:

```text
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib
```

Do not use the Cloud Run-oriented
`/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat:/usr/local/cuda/lib64`
path on the current Deep Learning VM. That path caused the container to load a
driver shim that mismatched the host NVIDIA 580 driver and forced a fail-closed
CUDA initialization error.

The worker is healthy only when `/health` reports:

```text
status = ok
served_backend = cuda
runtime.backend = cuda
runtime.execution_engine = rust_candle_csm_cuda
runtime.residency = warm_cuda
runtime.gpu_model = nvidia-l4-gce
runtime.refusal = null
```

2026-05-10 update:

- active image:
  `us-central1-docker.pkg.dev/openagents-lyra/lyra/psionic-csm-speech:211f4ca0`
- runtime health: `served_backend=cuda`, `runtime.residency=warm_cuda`,
  `runtime.execution_engine=rust_candle_csm_cuda`,
  `runtime.gpu_model=nvidia-l4-gce`, `runtime.refusal=null`
- startup metadata was updated to restart this image with the host-driver
  `LD_LIBRARY_PATH` shown above rather than the Cloud Run CUDA compatibility
  path.
- direct production speech smoke with `context_policy=prompt_profile_only` and
  `max_audio_length_ms=160` returned `audio/wav`, 7724 bytes,
  `x-psionic-generation-backend=cuda`,
  `x-psionic-generation-execution-engine=rust_candle_csm_cuda`,
  `x-psionic-gpu-model=nvidia-l4-gce`, and
  `x-psionic-csm-runtime-image-ref=...:211f4ca0`.
- direct production JSONL streaming smoke with `stream=true`,
  `stream_format=jsonl_base64`, and `max_audio_length_ms=2000` returned 4
  `audio` events plus 1 terminal event before the worker was reconnected to
  Autopilot production.

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
cargo run -p psionic-csm-speech --bin psionic-csm-speech-server -- --host 127.0.0.1 --port 8081
```

Useful environment controls:

- `PSIONIC_CSM_RUNTIME=disabled` starts the API in fail-closed publication mode
  without loading gated artifacts.
- `PSIONIC_CSM_BACKEND=cpu` runs the portable warm CPU backend.
- `PSIONIC_CSM_BACKEND=cuda` or `cuda:<ordinal>` runs the CUDA backend and
  fails closed if the CUDA device cannot initialize unless explicit CPU
  fallback is enabled.
- `PSIONIC_CSM_BACKEND=metal` or `metal:<ordinal>` is available for local
  Apple-silicon experiments but is not the production Cloud Run path.
- `PSIONIC_CSM_STARTUP_LOAD_MODE=sync|background` controls whether CSM/Mimi
  artifacts load before the server starts accepting traffic or in a background
  loader after the HTTP server is bound. CPU/local runs default to `sync`.
  CUDA Cloud Run deployments default to `background` so startup probes see a
  live process while `/health` reports `runtime.state=loading`.
- `PSIONIC_CSM_CPU_FALLBACK_ON_ACCELERATOR_FAILURE=true` permits explicit CPU
  fallback after accelerator initialization or load failure. It is off by
  default; GPU releases should keep it off.
- `PSIONIC_CSM_GPU_MODEL` publishes the expected accelerator class, such as
  `nvidia-l4`.
- `PSIONIC_CSM_CUDA_COMPUTE_CAP` defaults to `89` for L4 so Cloud Build can
  compile CUDA kernels without `nvidia-smi` in the builder.
- Cloud Run CUDA runtime images must expose `LD_LIBRARY_PATH` with NVIDIA
  driver mount paths before CUDA toolkit paths:
  `/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat:/usr/local/cuda/lib64`.
  Cloud Run GPU provides `libcuda.so.1` through the NVIDIA driver mount, not
  the CUDA toolkit directory. The runtime image also installs
  `cuda-compat-12-4` so the process can report a fail-closed health state
  instead of failing before application logs if the driver mount is absent.
- GCE Deep Learning VM containers should prefer the host driver path described
  in Current Production GPU Worker instead of the Cloud Run compat path.
- `PSIONIC_CSM_HOST`, `PSIONIC_CSM_PORT`, and `PSIONIC_CSM_MODEL_ID` mirror the
  command-line host, port, and model controls.
- `PSIONIC_CSM_RUNTIME_IMAGE_REF` optionally records the deploy image or source
  snapshot for governance publication. Local runs default to
  `not_configured_local_runtime`.

The current endpoints are:

- `GET /health`
- `GET /v1/models`
- `GET /psionic/csm/worker/metadata`
- `POST /v1/audio/speech`
- `POST /psionic/csm/speech`

The request shape accepts:

- `request_id`, the idempotency key Autopilot should reuse for retry-safe
  worker calls
- `model`, defaulting to `sesame/csm-1b`
- `input`
- `artifact_id`, optional, which must match the loaded CSM artifact id or CSM
  model digest
- `voice` or `voice_profile_id`
- `response_format`, currently only `wav`
- `stream`, returning generation-time audio chunks when `true`
- `stream_format`, optional, `multipart_mixed` or `jsonl_base64`; Autopilot uses
  `jsonl_base64` so its gateway can parse and forward chunks incrementally
- `cancellation_id`, optional, currently admitted as trace metadata while
  in-flight Rust CPU generation remains non-preemptible
- `timeout_ms`, defaulting to 10000 and capped at 30000
- `psionic_csm.temperature`
- `psionic_csm.top_k`
- `psionic_csm.max_audio_length_ms`, currently `80..=20000` on warm CPU/GPU.
  Autopilot production currently uses a 15000 ms budget for bounded generic
  spoken answers while CRM/action acknowledgements remain shorter.
- `psionic_csm.context_policy`, currently `prompt_profile_only` for governed
  Autopilot speech so CSM receives the approved source prompt codebooks for
  stable speaker identity; `none` remains available for diagnostics only

If no voice is provided, the server defaults to `openagents/default_female_v1`.
Requests for raw fixture prompt ids such as `conversational_a` or
`conversational_b` are refused because served requests must use governed
OpenAgents profile ids.

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
`x-psionic-model-id`, `x-psionic-request-id`, `x-psionic-cancellation-id`,
`x-psionic-timeout-ms`, `x-psionic-csm-artifact-id`,
`x-psionic-csm-governance-schema`, `x-psionic-csm-license-posture`,
`x-psionic-csm-runtime-image-ref`, `x-psionic-execution-engine`,
`x-psionic-csm-voice-profile-id`,
`x-psionic-csm-voice-approval-status`,
`x-psionic-csm-runtime-admission`, `x-psionic-csm-consent-posture`,
`x-psionic-csm-watermarking`, `x-psionic-csm-watermark-refusal-code`,
`x-psionic-csm-promotion-gate`, `x-psionic-csm-rollback-target`, CSM artifact
digest headers,
`x-psionic-first-audio-latency-ms`,
`x-psionic-full-generation-latency-ms`, `x-psionic-output-duration-ms`,
`x-psionic-csm-frames-sha256`, and `x-psionic-csm-wav-pcm16-digest`.

## Worker RPC Boundary

Autopilot should treat `POST /psionic/csm/speech` as the Psionic worker RPC
surface and `GET /psionic/csm/worker/metadata` as the worker contract surface.
The OpenAI-compatible `POST /v1/audio/speech` route remains compatibility
surface only.

Worker request fields:

```text
request_id
input
voice_profile_id
artifact_id
max_audio_length_ms
timeout_ms
cancellation_id
stream
```

Worker response metadata fields:

```text
request_id
cancellation_id
artifact_id
voice_profile_id
generated_frame_count
first_audio_latency_ms
full_generation_latency_ms
output_duration_ms
wav_pcm16_digest
codebook_frames_sha256
chunk_count
```

Worker governance metadata fields:

```text
artifact_id
artifact_hash
license_posture
runtime_image_ref
voice_profile_id
watermark_status
rollback_target
```

Current worker metrics:

```text
queue_depth
in_flight_requests
first_audio_latency_ms
full_generation_latency_ms
output_duration_ms
failure_code
runtime_state
```

The current Rust CPU generator is serialized through the resident runtime
lock. `cancellation_id` is therefore trace metadata and admission control for
now, not preemptive cancellation of a running Candle generation. Autopilot
must still be prepared to abandon stale responses client-side and fall back to
the current TTS provider when the worker exceeds its own timeout.

Provider output is evidence, not instruction. This RPC boundary carries no
browser-facing UI authority, CRM authority, Blueprint authority, or product
routing authority.

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
  `unsupported_operator_accepted_limited_dogfood` with
  `csm_watermarking_unavailable`, so CSM output is admitted only for
  OpenAgents-operated Autopilot dogfood and remains unavailable for arbitrary
  public voice cloning
- artifact governance truth: license posture, runtime image ref, quantization,
  allowed voice profile ids, disallowed voice use cases, canary/primary
  promotion gate, missing governance blocks, and rollback target
- runtime truth: `ready`/`unavailable`, warm-load latency, backend,
  residency, artifact availability, and accelerated-backend refusal truth

One-shot local request:

```bash
curl -D /tmp/psionic-csm-wav.headers \
  -o /tmp/psionic-csm.wav \
  -X POST http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"sesame/csm-1b","input":"hello from psionic","voice_profile_id":"openagents/default_female_v1","response_format":"wav","psionic_csm":{"max_audio_length_ms":160,"context_policy":"prompt_profile_only"}}'
```

Generation-time multipart stream request:

```bash
curl -D /tmp/psionic-csm-stream.headers \
  -o /tmp/psionic-csm.multipart \
  -X POST http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"sesame/csm-1b","input":"hello from psionic","voice_profile_id":"openagents/default_female_v1","response_format":"wav","stream":true,"psionic_csm":{"max_audio_length_ms":160,"context_policy":"prompt_profile_only"}}'
```

Generation-time JSONL stream request for Autopilot gateway consumption:

```bash
curl -N -D /tmp/psionic-csm-jsonl.headers \
  -X POST http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"sesame/csm-1b","input":"hello from psionic","voice_profile_id":"openagents/default_female_v1","response_format":"wav","stream":true,"stream_format":"jsonl_base64","psionic_csm":{"max_audio_length_ms":160,"context_policy":"prompt_profile_only"}}'
```

## Cloud Run Deployment

The repeatable production deployment entrypoint is:

```bash
scripts/deploy-csm-speech-cloud-run.sh
```

Default deployment settings:

- project: `openagents-lyra`
- region: `us-central1`
- service: `psionic-csm-speech`
- current public service URL:
  `https://psionic-csm-speech-ycgawzh3ta-uc.a.run.app`
- image:
  `us-central1-docker.pkg.dev/openagents-lyra/lyra/psionic-csm-speech:<git-sha>`
- service account:
  `psionic-csm-speech@openagents-lyra.iam.gserviceaccount.com`
- private artifact bucket:
  `openagents-lyra-psionic-csm-artifacts`
- Cloud Run CPU shape: 8 CPU, 32 GiB memory, min 1, max 3, concurrency 1,
  CPU always allocated
- Cloud Run GPU shape when `PSIONIC_CSM_BACKEND=cuda`: 1 `nvidia-l4` GPU by
  default, gen2 execution, GPU zonal redundancy disabled by default for faster
  availability, and no CPU fallback unless explicitly enabled
- artifact mount:
  `gs://openagents-lyra-psionic-csm-artifacts` mounted read-only at
  `/root/.cache/huggingface`
- runtime env:
  `HF_HOME=/root/.cache/huggingface`,
  `PSIONIC_CSM_RUNTIME=true`,
  `PSIONIC_CSM_BACKEND=cpu` or `cuda`,
  `PSIONIC_CSM_STARTUP_LOAD_MODE=sync` for CPU or `background` for CUDA,
  `PSIONIC_CSM_GPU_MODEL=nvidia-l4`,
  `PSIONIC_CSM_CPU_FALLBACK_ON_ACCELERATOR_FAILURE=false`, and
  `PSIONIC_CSM_MODEL_ID=sesame/csm-1b`
- CUDA runtime env also sets
  `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat:/usr/local/cuda/lib64`,
  `NVIDIA_VISIBLE_DEVICES=all`, and `NVIDIA_DRIVER_CAPABILITIES=compute` so the
  Cloud Run GPU driver-provided `libcuda.so.1` is visible before process start.

The script stages only the minimal gated Hugging Face cache objects required by
the Rust server:

- CSM config and model blobs for `sesame/csm-1b`
- Llama tokenizer snapshot file for `meta-llama/Llama-3.2-1B`
- Mimi safetensors blob for `kyutai/moshiko-pytorch-bf16`

No Hugging Face token, provider key, Python virtualenv, Python CSM repo, or raw
prompt audio is uploaded by this deploy path. The script finishes only after
`/health` reports `runtime.state=ready` and a production
`POST /v1/audio/speech` smoke returns a non-empty WAV.

GPU deployment uses:

```bash
PSIONIC_CSM_BACKEND=cuda scripts/deploy-csm-speech-cloud-run.sh
```

The script builds with the `csm-cuda` Cargo feature in a CUDA devel image,
deploys a tagged no-traffic Cloud Run candidate revision with GPU flags, waits
on the candidate tag URL, runs the speech smoke against that candidate, and
only then promotes the exact tested revision to 100% traffic. The release is
blocked if the ready candidate does not publish `served_backend = cuda`.

GPU startup is two-phase. The container binds the HTTP server first, publishes
`runtime.state=loading`, and then hydrates the tokenizer, CSM weights, and Mimi
decoder in a background loader. The release script still waits for
`runtime.state=ready` before running the speech smoke, so a deployment is not
considered promoted until the real CUDA-backed model is loaded and serving.

2026-05-06 production deployment evidence:

- build id: `cbd3ec86-0a68-4f1d-b254-00882eb9f9b2`
- revision: `psionic-csm-speech-00001-jtd`
- image digest:
  `sha256:d17850fab527252cefa26f48a4a46ffd10855a89bd578094c17cbf46e8c55c3c`
- `/health`: `status=ok`, `runtime.state=ready`, `residency=warm_cpu`,
  `execution_engine=rust_candle_csm_cpu`, `load_latency_ms=146371`,
  `accelerated_backend=unavailable_fail_closed`
- speech smoke: `POST /v1/audio/speech` returned `audio/wav`, 7724 bytes,
  160 ms output, 2 generated CSM frames, `full_generation_latency_ms=5128`,
  governed voice profile `openagents/default_female_v1`, and watermarking posture
  `unsupported_operator_accepted_limited_dogfood`
- historical downstream production smoke: the retired voice surface called this
  service through a Psionic TTS provider setting; public one-shot and `/audio`
  smokes returned `audio/wav` assistant audio through the old voice domain.

## Voice Profile Governance

The served voice-profile contract is now:

- public served id: `openagents/default_female_v1`
- source prompt profile: `conversational_b`
- approval status: `approved_openagents_operated_dogfood`
- runtime admission:
  `admitted_openagents_operated_autopilot_production_dogfood`
- allowed surfaces: `psionic_local_development`,
  `autopilot_internal_development`, and `autopilot_production_dogfood`
- disallowed surfaces: `public_user_voice_clone` and
  `arbitrary_reference_audio_upload`
- source provenance:
  `committed_csm_parity_fixture_prompt_with_full_precomputed_codebooks`
- consent posture:
  `openagents_operated_placeholder_from_committed_reference_prompt_not_arbitrary_user_upload`

The route refuses unknown or ungoverned profile ids with
`voice_profile_unavailable`. This is intentional: prompt fixtures are not
served voice ids. A source prompt can feed a governed profile, but callers must
use the governed OpenAgents profile id.

Arbitrary voice cloning and reference-audio upload are out of scope until a
consent system exists. The current Rust Mimi encode capability remains refused
with `rust_mimi_encode_not_implemented`, and CSM watermarking remains
`unsupported_operator_accepted_limited_dogfood` with
`csm_watermarking_unavailable`.

Public demo watermark keys are not production safety controls. OpenAgents has
accepted this profile only for its own bounded OpenAgents production dogfood. Public
voice cloning, arbitrary reference-audio upload, and broader user-selectable
CSM voices remain blocked until Psionic has a private watermark or equivalent
voice-safety control.

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
keeps that CPU path warm and serves short WAV responses plus generation-time
multipart or JSONL chunks. It is still correctness-first rather than
production-latency-ready. Metal/CUDA acceleration, smaller chunk tuning,
preemptive cancellation, prompt-codebook context use, production voice consent,
and production watermark posture remain separate cutover gates.

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
- generation-time multipart stream returning `200` and
  `multipart/mixed; boundary=psionic-csm-stream`
- output duration: `160 ms`
- generated CSM frame count: `2`
- one-shot full-generation latency: `2702 ms`
- stream full-generation latency: `2677 ms`

The voice-governance smoke records:

- `/health` publishing `openagents/default_female_v1`
- source prompt mapping: `conversational_b`
- raw `conversational_b` speech requests refused with
  `voice_profile_unavailable`
- governed `openagents/default_female_v1` speech request returning `200 audio/wav`
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

Run the focused production-server crate tests with:

```bash
cargo test -p psionic-csm-speech csm_
```

Run the direct generation-time streaming smoke against the active worker with:

```bash
node scripts/csm-streaming-smoke.mjs
```

That smoke requires at least two JSONL `audio` events, one terminal event, and a
first audio event before response completion, so it fails if the endpoint buffers
a complete WAV/body before yielding the first audio chunk.
The production streaming window is currently 2 generated CSM frames per emitted
audio chunk so Autopilot receives roughly 160 ms windows instead of the earlier
8-frame/640 ms cadence. A 1-frame window was tested but rolled back because the
integrated Autopilot gateway path timed out waiting for assistant audio.

The validator checks:

- fixture schema and artifact digest shapes
- required prompt profile ids
- prompt WAV metadata
- tokenizer frame dimensions and text-lane mask semantics
- full committed Mimi prompt-codebook dimensions and token bounds for governed
  `prompt_profile_only` speech context
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
- generation-time multipart stream framing with ordered binary WAV chunks and
  terminal metadata
- JSONL base64 stream framing for Autopilot gateway consumption
- governed OpenAgents voice-profile admission and raw fixture prompt refusal
- explicit fixture-gap truth for exact deterministic prompted replay
- explicit refusal truth for runtime reference-audio encoding

## Next Phases

The phase sequence lives in GitHub under `OpenAgentsInc/psionic#959`.

Next work:

1. Keep OpenAgents production dogfood on the Psionic TTS provider boundary and
   record provider route, latency, and refusal evidence on each release.
2. Add Metal/CUDA acceleration and true frame-by-frame low-latency decode when
   CSM quality and voice governance justify product cutover work.
3. Add private watermark or equivalent voice-safety controls before any public
   voice-cloning or arbitrary reference-audio feature.
