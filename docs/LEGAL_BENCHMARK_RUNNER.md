# Legal Benchmark Agent Runner

> Status: implemented_early.

The Rust legal benchmark runner lives in
`crates/psionic-eval/src/legal_benchmark_agent.rs`. It connects the provider
adapter layer to the closed benchmark tool surface and writes replayable run
artifacts for evaluator and Autopilot4 import.

## Loop

The runner:

- builds a system prompt from run policy, allowed tools, and module
  instructions
- sends task instructions, deliverables, and criteria as the first user
  message
- calls a provider-neutral `ModelAdapter`
- records every model response as a transcript event
- executes model tool calls through `execute_legal_benchmark_tool`
- returns tool results to the model with provider tool-call ids preserved
- accepts explicit finalization only when the assistant returns JSON shaped as
  `{"action":"submit","deliverables":["path"]}` or
  `{"action":"finalize","deliverables":["path"]}`

## Terminal States

The runner maps stop and failure conditions into `RunTerminalState`:

- `submitted`
- `no_tool_calls`
- `max_turns`
- `max_tokens`
- `context_overflow`
- `provider_failure`
- `sandbox_failure`
- `policy_failure`
- `internal_error`

Tool failures are transcripted. Sandbox unavailability or sandbox failure ends
the run as `sandbox_failure`; other tool errors can be returned to the model in
the next turn.

## Run Directory

Each run directory contains:

- `config.json`
- `transcript.jsonl`
- `metrics.json`
- `output_artifact_manifest.json`
- `extraction_receipts.json`
- `tool_receipts.json`
- `run_record.json`
- `run_receipt.json`

Run ids include task id, run-config hash prefix, and a caller nonce when one is
provided. This keeps repeated sweeps distinguishable while preserving stable
traceability to task/config identity.

## Receipts

`LegalBenchmarkRunReceipt` binds:

- task spec hash
- input artifact manifest hash
- run config hash
- output artifact manifest hash
- transcript hash
- metrics hash
- tool receipts hash
- run record hash

The evaluator should consume `run_record.json`,
`output_artifact_manifest.json`, and `run_receipt.json` together rather than
trusting a loose output folder.

## Fixture

`fixtures/legal_benchmark/agent_run_mock/` contains a small checked example of
the transcript and run receipt shape produced by a deterministic mock-provider
run. Unit tests exercise the same path end to end in a temporary directory and
verify output files, transcript, metrics, tool receipts, run record, and run
receipt are written.
