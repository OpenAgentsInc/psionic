# CSM Speech Benchmarks

## Purpose

This is the release evidence path for Psionic CSM speech generation used by
Autopilot Voice. It measures whether the served worker is actually using the
claimed backend, whether it silently fell back to CPU, and whether short
acknowledgements are fast enough for operator dogfood.

## Canonical Command

Run against a local or deployed CSM speech worker:

```sh
node scripts/csm-speech-benchmark.mjs \
  --url https://psionic-csm-speech-ycgawzh3ta-uc.a.run.app \
  --repeat 3
```

The script reads
`fixtures/csm/benchmarks/csm_speech_benchmark_corpus.v1.json` and writes a
machine-readable report under `target/csm-benchmarks/`.

## Metrics

The report records:

- service health: `served_backend`, `execution_engine`, `runtime`;
- response headers: served backend, generation backend, execution engines,
  accelerated backend, GPU model, CPU fallback reason;
- wall latency, first-audio latency, full-generation latency, output duration,
  WAV byte size, and status;
- P50/P95 summaries for first-audio and full-generation latency;
- success, failure, and explicit CPU fallback counts.

## Promotion Thresholds

GPU-backed CSM may be promoted as the Autopilot primary TTS path only when:

- `/health` reports `served_backend = cuda`, `runtime.backend = cuda`,
  `runtime.execution_engine = rust_candle_csm_cuda`, and
  `runtime.accelerated_backend = cuda`;
- benchmark responses publish `x-psionic-served-backend: cuda` and
  `x-psionic-accelerated-backend: cuda`;
- `cpu_fallback_count = 0` unless the release explicitly declares a fallback
  drill;
- the benchmark corpus has `success_rate = 1.0`;
- short acknowledgement P50 full-generation latency is materially below the
  old CPU path and should target sub-2s for normal product use;
- P95 full-generation latency remains under 5s before broad internal dogfood;
- every generated payload is browser-playable WAV and includes backend evidence
  headers.

CPU fallback is allowed only when
`PSIONIC_CSM_CPU_FALLBACK_ON_ACCELERATOR_FAILURE=true` is deliberately set.
Fallback responses must publish `x-psionic-cpu-fallback-reason` and must not be
reported as GPU.

## Failure Evidence

Failures are release-blocking when:

- a CUDA deployment becomes ready as `served_backend=cpu`;
- a CUDA deployment omits GPU or accelerated-backend headers;
- the worker succeeds but returns non-WAV bytes;
- CSM/Mimi load failure is hidden behind a generic success response;
- benchmark output cannot distinguish true GPU execution from CPU fallback.
