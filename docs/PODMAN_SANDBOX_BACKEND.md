# Podman Sandbox Backend

> Status: implemented_early for local command construction, path safety, and
> structured command receipts.

Psionic's legal benchmark runner needs a host-managed sandbox before document
extractors and agent tools can run against the Harvey-compatible task corpus.
The first Rust backend lives in `crates/psionic-sandbox/src/podman.rs`.

## Contract

The backend accepts a `PodmanSandboxConfig` with:

- container image
- canonical workdir
- bind mounts
- environment variables
- CPU, memory, PID, and timeout limits
- network policy
- allowed host roots
- read-only root filesystem posture

The legal benchmark helper mounts:

- source artifacts at `/workspace/inputs` as read-only
- scratch files at `/workspace/scratch` as read-write
- runner outputs at `/workspace/output` as read-write

Network access is disabled by default through `PodmanNetworkPolicy::Disabled`.
Any bridge or host networking exception must be expressed explicitly in the
config and will be visible in the resulting command receipt.

## Path Safety

The backend resolves allowed host roots and mount paths before it builds the
Podman command. It rejects:

- empty images or commands
- non-absolute container paths
- container root mounts
- `.` or `..` components in container paths
- duplicate container mount points
- missing host paths
- host paths containing mount separators
- host paths that canonicalize outside the declared allowed roots

The symlink check uses the canonical target path, so a symlink inside an
allowed directory that points outside that directory is rejected before
execution.

## Receipts

`run_prepared_sandbox_command` captures:

- stdout and stderr bytes
- stdout and stderr SHA-256 digests
- exit code
- Unix signal when available
- timeout state
- wall time
- configured resource limits
- image and network policy

Spawn, poll, and output collection failures remain typed host-side errors.
Completed commands, non-zero exits, and timeouts return structured receipts so
the benchmark runner can preserve failure evidence.

## Fixture

The checked fixture is:

- `fixtures/legal_benchmark/podman_sandbox_config.json`

It documents the intended production shape for the legal benchmark sandbox and
is parsed by unit tests.

## Current Limits

This backend builds and runs local Podman commands, but it does not yet own:

- remote sandbox execution
- Cloud Run sandbox adapters
- cgroup metric collection beyond configured limits
- document-extractor-specific receipt binding

Those are expected follow-on layers for the legal benchmark runner.
