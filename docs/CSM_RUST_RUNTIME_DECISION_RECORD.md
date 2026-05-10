# CSM Rust Runtime Decision Record

Status: accepted for Psionic CSM worker evolution

Date: 2026-05-09

Owner: Psionic

Related spec: [CSM Audio Runtime](CSM_AUDIO_RUNTIME.md)

## Decision

Psionic will move CSM speech generation toward a Rust-owned production runtime.
Python demo code remains an offline reference, fixture generator, and parity
oracle only. It is not the product architecture, it is not a sidecar worker,
and it is not admitted into the serving path.

The current admitted runtime is:

- Rust HTTP worker surface in `psionic-serve`;
- Rust tokenizer, prompt framing, artifact descriptor, and voice-profile
  governance in `psionic-models`;
- Rust Candle CSM generation with CPU as the portable fallback and CUDA as the
  admitted Cloud Run GPU backend;
- Rust `moshi` Mimi decode to 24 kHz mono PCM16 WAV on the same requested
  backend when supported;
- warm CPU or warm CUDA residency, one request at a time, fail-closed
  unsupported backend publication;
- worker metadata, request id, artifact id, timeout, cancellation id, latency,
  digest, and refusal metadata for Autopilot shadow/canary evaluation.

The target runtime remains Rust-owned even when implementation details change.
Acceleration, batching, streaming, scheduler control, model loading, artifact
verification, voice governance, and telemetry all stay inside Psionic-owned
Rust services or Rust-hosted workers.

Provider output is evidence, not instruction. CSM does not own Autopilot
conversation state, HUD routing, CRM decisions, Blueprint Program Runs,
Action Submissions, approvals, evidence, or receipts.

## Production Interface

The production-facing worker contract is:

```text
POST /psionic/csm/speech
GET  /psionic/csm/worker/metadata
```

The OpenAI-compatible route remains a compatibility surface:

```text
POST /v1/audio/speech
```

Autopilot should depend on the Psionic worker route and the worker metadata
surface, not on implementation internals. The worker request must carry:

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

The response must expose:

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
failure_code when failed
runtime_state
```

The worker may add fields over time, but it must preserve enough metadata for
Autopilot to evaluate provider quality by safe business outcome, not only by
audio presence or transcript similarity.

## Runtime Options Evaluated

| Option | Decision | Reason |
| --- | --- | --- |
| Python `run_csm.py` demo wrapper | Rejected for production | Useful as reference fixture input, but it would make Python environment, Torch/Moshi package state, local repo layout, and demo assumptions part of the product runtime. |
| Python sidecar called by Rust | Rejected for production | It hides latency, cancellation, artifact verification, tokenizer, voice governance, and telemetry behind a process seam that Psionic cannot fully own. |
| Rust Candle CPU CSM + Rust Mimi decode | Admitted fallback worker MVP | It proves native Rust artifact loading, generation, decode, headers, refusals, and Autopilot worker metadata. It is correctness-first and latency-limited. |
| Rust Candle CUDA CSM + Rust Mimi decode | Admitted Cloud Run GPU lane | Keeps model execution in Rust while reducing full-generation latency. It must publish CUDA backend evidence and fail closed unless explicit CPU fallback is enabled. |
| Rust Candle Metal CSM + Rust Mimi decode | Local experiment lane | Useful for Apple-silicon development, but not the production Cloud Run path. |
| Full Psionic-native compiled runtime | Long-term target | Best fit for scheduler, batching, graph ownership, fusion, and cross-hardware execution. It should follow the verified Rust worker contract rather than replace it with a new product API. |
| Third-party hosted TTS or OpenAI TTS | Baseline/fallback only | Useful as quality and latency baseline while CSM matures. It must not become architectural authority. |

## Tokenizer Boundary

Tokenizer ownership belongs in Rust.

The current admitted path loads the Llama tokenizer from a cached Hugging Face
`tokenizer.json` using the Rust `tokenizers` crate. Prompt text, speaker tags,
BOS/EOS template installation, 33-lane text/audio frame construction, and
prompt-window validation happen before generation in Rust.

The Python tokenizer path is useful only as parity evidence. It must not be
called by the serving worker.

## Mimi And Audio-Code Boundary

Mimi decode is admitted in Rust through the `moshi` crate.

Current capability:

- load Kyutai Mimi safetensors;
- validate expected weight digest;
- decode 32-codebook RVQ frames;
- remove trailing all-zero EOS frames;
- output PCM16 WAV bytes suitable for browser playback.

Current refusal:

- runtime reference-audio encoding is not implemented in Rust;
- arbitrary uploaded reference audio is refused;
- ungoverned voice ids are refused;
- watermarking is unavailable and blocks broader public promotion.

The next admitted encode path must be Rust-owned. It may harvest algorithmic
details from reference implementations, but production encode cannot depend on
Python runtime packages.

## Batching, Scheduling, And Cancellation

The current CPU worker is serialized behind the resident runtime lock.
`cancellation_id` is therefore trace and admission metadata, not preemptive
cancellation once Candle generation is running.

The next production scheduler must add:

- explicit queue depth and in-flight request publication;
- request admission before model execution starts;
- stale request abandonment by `request_id` and `cancellation_id`;
- server-side timeout enforcement around admission and generation;
- bounded max audio length by deployment tier;
- per-worker concurrency configuration;
- clear fallback signal when generation exceeds Autopilot's voice turn budget.

Preemptive cancellation during compute is a target for the accelerated worker
lane, not a current CPU guarantee.

## Quantization, GPU, And Memory Profile

The fallback worker is warm CPU. Existing smoke evidence shows correctness with
multi-second generation:

- retained local one-shot smoke: about `2702 ms` full generation for 160 ms of
  output;
- retained production smoke: about `5128 ms` full generation for 160 ms of
  output;
- startup warm load on the retained production evidence: about `146371 ms`.

Those measurements are useful for correctness and deployment proof, not final
product latency.

The GPU worker must report:

- model artifact id and digest;
- requested backend;
- served backend;
- execution engine;
- accelerated backend;
- GPU model;
- memory required to load CSM and Mimi together;
- first-audio latency;
- full-generation latency;
- output duration;
- generated frame count;
- GPU or accelerator utilization where available;
- fallback/refusal state when accelerator artifacts or devices are absent.

CUDA releases use
`PSIONIC_CSM_BACKEND=cuda scripts/deploy-csm-speech-cloud-run.sh`. The deploy
blocks if `/health` becomes ready as any backend other than CUDA.

CUDA Cloud Run startup uses `PSIONIC_CSM_STARTUP_LOAD_MODE=background`. The
server binds first and reports `runtime.state=loading` while the background
loader hydrates CSM and Mimi on the requested accelerator. This keeps Cloud Run
startup probing separate from model-load latency without weakening promotion:
release automation still waits for `runtime.state=ready`, `served_backend=cuda`,
and CUDA response headers before admitting the revision.

The CUDA runtime image must also keep the Cloud Run GPU driver path in
`LD_LIBRARY_PATH`:
`/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat:/usr/local/cuda/lib64`.
The CUDA toolkit runtime image contains cuBLAS/cuRAND, but Cloud Run exposes
`libcuda.so.1` through the NVIDIA driver mount. Missing that path prevents the
Rust binary from starting before any application logs can be emitted. The
runtime image includes `cuda-compat-12-4` as a secondary compatibility path,
with the real Cloud Run driver path still ordered first.

Quantization is allowed only if it preserves artifact identity, fixture
comparability, and promotion evidence. A quantized CSM artifact must publish a
new artifact id and cannot silently replace the full-precision artifact.

## Autopilot Shadow And Canary Dependencies

Autopilot may shadow or canary Psionic CSM only when these surfaces exist and
stay current:

- `/psionic/csm/worker/metadata`;
- stable request id and artifact id handling;
- governed `voice_profile_id`;
- response headers or metadata for latency, generated frame count, output
  duration, WAV digest, codebook-frame digest, and refusal code;
- fail-closed unsupported backend and missing artifact behavior;
- artifact and license governance metadata;
- clear fallback posture to the current production TTS provider;
- replayable voice turn evidence outside raw private audio retention.

Autopilot owns voice session state, VAD, endpointing, text response
composition, HUD handoff, user-facing playback, and fallback routing. Psionic
owns the CSM worker runtime and execution evidence.

## Risks

- Warm CPU latency is too high for primary conversation TTS.
- Lack of preemptive cancellation can waste compute after the browser abandons
  a turn.
- Missing Rust Mimi encode prevents arbitrary reference-audio profiles and
  exact prompted replay from committed prompt fixtures.
- Missing watermarking blocks public voice-cloning and broad user-selectable
  CSM voices.
- GPU acceleration can introduce new artifact, memory, determinism, and
  fallback failure modes.

## Next Implementation Steps

1. Complete CSM artifact, license, voice-profile, and watermark governance so
   missing governance blocks canary/primary promotion.
2. Add a worker smoke that exercises `/psionic/csm/worker/metadata`, disabled
   runtime failure, artifact mismatch failure, and a governed short speech
   request when gated artifacts are present.
3. Add scheduler/backpressure state around the CPU worker before increasing
   concurrency.
4. Keep the CUDA Cloud Run lane benchmarked with
   `scripts/csm-speech-benchmark.mjs` and promote only with backend headers,
   no silent CPU fallback, and documented latency evidence.
5. Add first-audio streaming design for real chunks, not only buffered
   multipart WAV chunks.
6. Add Rust Mimi encode or an equivalent governed profile-building path before
   enabling arbitrary reference audio.
7. Feed Autopilot shadow/canary benchmark results back into this decision
   record with business-outcome, latency, fallback, and playback metrics.

## Non-Goals

- Do not ship Python demo code as the serving architecture.
- Do not hide the runtime behind an opaque subprocess without Psionic artifact,
  telemetry, and refusal control.
- Do not make CSM output business authority.
- Do not allow ungoverned voice cloning or arbitrary reference audio upload.
- Do not treat one manual good sample as promotion evidence.
