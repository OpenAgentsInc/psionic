# Local Metal Benchmark Desktop Starvation Audit

## Incident

On April 7, 2026, three full `gemma4_bench` Metal runs were launched in parallel on the active desktop Mac:

- `PSIONIC_METAL_DECODE_SIMDGROUPS=4 ./target/release/examples/gemma4_bench ...`
- `PSIONIC_METAL_DECODE_SIMDGROUPS=8 ./target/release/examples/gemma4_bench ...`
- `PSIONIC_METAL_DECODE_SIMDGROUPS=16 ./target/release/examples/gemma4_bench ...`

The user reported that the machine became unresponsive for about ten seconds.

## Evidence

- The benchmarked GGUF is `8.9G`:
  - `/Users/christopherdavid/models/gemma4/gemma4-e4b-ollama.gguf`
- The host has `137438953472` bytes of RAM, which is `128 GiB`.
- After the incident there were no surviving `gemma4_bench` processes.
- `ollama serve` and one `ollama runner` remained resident.
- An older `psionic-openai-server` process remained resident on CPU with `0.0` CPU and about `0.7%` memory, so it was not the main cause of the stall.

## Root Cause

The stall was caused by benchmark scheduling, not by a mysterious background bug.

Each `gemma4_bench` Metal run does all of the following:

- opens the full `8.9G` GGUF
- allocates its own runtime buffers
- schedules long decode loops on the local Metal device
- competes for Apple Silicon unified-memory bandwidth

Launching three of those runs in parallel on the active desktop means the window server, browser, and the rest of the UI must compete with three independent model loads and three independent GPU decode streams. On an interactive Mac, that is enough to cause visible desktop starvation.

## Prevention

The repo now prevents the exact failure mode that caused this incident.

- `crates/psionic-serve/examples/gemma4_bench.rs` now acquires a host-local lock before any run that uses the local Metal path.
- A second concurrent local Metal launch now fails immediately with a clear error instead of silently starting another heavy run.
- The override is explicit:
  - `PSIONIC_ALLOW_PARALLEL_METAL_BENCH=1`

## Operating Rule

For interactive macOS hosts:

- never run parallel full-model Metal benchmarks
- never use `multi_tool_use.parallel` for local Metal benchmark sweeps
- run candidate variants serially
- prefer a remote Tailnet node for comparative throughput sweeps
- treat Ollama and other live local model runners as competing workloads unless the host is intentionally dedicated to benchmarking

This rule is also recorded in `psionic/AGENTS.md` so future agent sessions inherit it.
