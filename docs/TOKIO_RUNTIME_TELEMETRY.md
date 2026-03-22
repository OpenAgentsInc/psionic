# Tokio Runtime Telemetry

> Status: canonical issue `#313` Tokio-runtime telemetry contract, written.

## Why This Doc Exists

Psionic owns the serving and transport runtimes around model execution, not
just the inner model compute itself. This doc freezes how Psionic enables
`dial9-tokio-telemetry` on Tokio-owned serving and transport surfaces so
operators can separate runtime-side latency from backend-side compute latency.

The first admitted surfaces are:

- `crates/psionic-serve`
- `crates/psionic-mlx-serve`
- `crates/psionic-net`
- the shared runtime helper in `crates/psionic-observe`

## Current Scope

When enabled, Psionic now records Tokio runtime telemetry for:

- `psionic-openai-server`
- `psionic-gpt-oss-server`
- `psionic-mlx-serve serve`
- `psionic-net` transport and relay tasks when they run inside a runtime built
  through `psionic-observe`

The server path now uses a repo-owned Axum accept loop that routes per-
connection tasks through the dial9 `TelemetryHandle` when telemetry is active.
`psionic-net` now routes its long-lived transport and relay tasks through the
same helper instead of hard-wired `tokio::spawn`.

## Non-Goals

This lane does not claim to profile:

- inner compute on dedicated `std::thread` model workers
- llama.cpp subprocess internals
- CUDA, Metal, MLX, or GGUF backend kernels
- app-owned orchestration outside Psionic runtimes

It explains the Tokio side around those lanes: scheduling delay, async handler
work, queueing, and wait time.

## Build Gating

Dial9 requires Tokio unstable hooks. The supported build posture is:

- add `RUSTFLAGS='--cfg tokio_unstable'`
- enable `tokio-runtime-telemetry` on the runtime-owning crate
- optionally enable `tokio-runtime-telemetry-cpu` for Linux perf-backed CPU
  samples and sched-event capture

The default build remains a plain Tokio runtime. If telemetry is disabled,
Psionic installs no dial9 hooks and falls back to normal `tokio::spawn`.

## Runtime Configuration

Psionic uses one shared environment-driven config surface:

- `PSIONIC_TOKIO_TELEMETRY_ENABLED`
  Values: `1/0`, `true/false`, `yes/no`
- `PSIONIC_TOKIO_TELEMETRY_TRACE_PATH`
  Rotating trace path prefix. Required when telemetry is enabled.
- `PSIONIC_TOKIO_TELEMETRY_TASK_TRACKING`
  Enables task spawn/terminate metadata and traced spawn helpers.
- `PSIONIC_TOKIO_TELEMETRY_LINUX_CPU_PROFILING`
  Linux-only perf sampling request. Requires the `tokio-runtime-telemetry-cpu`
  feature.
- `PSIONIC_TOKIO_TELEMETRY_LINUX_SCHED_EVENTS`
  Linux-only scheduler-event capture request. Requires the
  `tokio-runtime-telemetry-cpu` feature.
- `PSIONIC_TOKIO_TELEMETRY_ROTATE_AFTER_BYTES`
  Per-segment trace size ceiling. Default: `16777216`.
- `PSIONIC_TOKIO_TELEMETRY_MAX_TOTAL_BYTES`
  Total retained trace size ceiling. Default: `268435456`.

## Operator Flows

### OpenAI-Compatible CPU Server

```bash
RUSTFLAGS='--cfg tokio_unstable' \
PSIONIC_TOKIO_TELEMETRY_ENABLED=1 \
PSIONIC_TOKIO_TELEMETRY_TRACE_PATH=/tmp/psionic/openai-trace.bin \
PSIONIC_TOKIO_TELEMETRY_TASK_TRACKING=1 \
cargo run -p psionic-serve --features tokio-runtime-telemetry --bin psionic-openai-server -- \
  -m /path/to/model.gguf
```

### GPT-OSS Server

```bash
RUSTFLAGS='--cfg tokio_unstable' \
PSIONIC_TOKIO_TELEMETRY_ENABLED=1 \
PSIONIC_TOKIO_TELEMETRY_TRACE_PATH=/tmp/psionic/gpt-oss-trace.bin \
PSIONIC_TOKIO_TELEMETRY_TASK_TRACKING=1 \
cargo run -p psionic-serve --features tokio-runtime-telemetry --bin psionic-gpt-oss-server -- \
  -m /path/to/model.gguf
```

### MLX Serve

```bash
RUSTFLAGS='--cfg tokio_unstable' \
PSIONIC_TOKIO_TELEMETRY_ENABLED=1 \
PSIONIC_TOKIO_TELEMETRY_TRACE_PATH=/tmp/psionic/mlx-trace.bin \
PSIONIC_TOKIO_TELEMETRY_TASK_TRACKING=1 \
cargo run -p psionic-mlx-serve --features tokio-runtime-telemetry --bin psionic-mlx-serve -- \
  serve --reference /path/to/model.gguf
```

### Linux CPU Sampling And Sched Events

```bash
RUSTFLAGS='--cfg tokio_unstable' \
PSIONIC_TOKIO_TELEMETRY_ENABLED=1 \
PSIONIC_TOKIO_TELEMETRY_TRACE_PATH=/tmp/psionic/openai-trace.bin \
PSIONIC_TOKIO_TELEMETRY_TASK_TRACKING=1 \
PSIONIC_TOKIO_TELEMETRY_LINUX_CPU_PROFILING=1 \
PSIONIC_TOKIO_TELEMETRY_LINUX_SCHED_EVENTS=1 \
cargo run -p psionic-serve --features tokio-runtime-telemetry-cpu --bin psionic-openai-server -- \
  -m /path/to/model.gguf
```

## Linux Caveats

The Linux CPU/scheduler lane depends on `perf_event_open` access and good stack
unwinding posture. Operators should expect:

- frame pointers materially improve stack quality
- `perf_event_paranoid` may block non-root sampling
- `kptr_restrict` can limit kernel symbol visibility
- non-Linux builds still emit poll/park/wake/queue telemetry, but not the
  Linux perf-backed extras

## What The Traces Mean

This lane is intended to answer questions like:

- was p99 inflation caused by Tokio worker scheduling delay?
- did the HTTP handler spend CPU time on the runtime before the request reached
  backend execution?
- did the request wait on async I/O or response-state work?
- was the async runtime idle while a dedicated worker thread or subprocess did
  the real compute?

Those are runtime-boundary questions. Backend-specific profiling remains
separate work.

## Transport Reuse

`psionic-net` does not own a standalone telemetry CLI. Instead it reuses the
same runtime helper:

- build the hosting Tokio runtime with `psionic_observe::build_main_runtime`
  or `psionic_observe::build_runtime`
- enable the same environment keys above
- transport and relay tasks spawned by `LocalClusterNode` and
  `ClusterRelayServer` will then inherit traced task spawning automatically
