# Compiled Agent Default Row Reference

> Status: canonical `AGENT-PLATFORM` default-row record, updated 2026-03-28.

## What This Closes

Psionic now owns one explicit default learned row for the first compiled-agent
loop instead of pretending every local model row is equivalent.

The canonical contract lives in:

- `crates/psionic-eval/src/compiled_agent_eval.rs`
- `crates/psionic-train/src/compiled_agent_learning.rs`
- `crates/psionic-train/src/bin/compiled_agent_default_row_contract.rs`
- `crates/psionic-train/src/bin/compiled_agent_default_row_probe.rs`
- `scripts/check-compiled-agent-default-row-contract.sh`
- `fixtures/compiled_agent/compiled_agent_default_row_v1.json`

## Default Row

The current default learned row is:

- row id: `compiled_agent.qwen35_9b_q4km.archlinux.consumer_gpu.v1`
- host: `archlinux`
- accelerator: `rtx_4080_16gb`
- model family: `qwen35`
- model artifact: `qwen3.5-9b-q4_k_m-registry.gguf`

This row is the honest target for the first compiled-agent loop. It is not a
claim that weaker rows are equivalent.

## Admitted Task Scope

This row is admitted for:

- `intent_route` for explicit provider-readiness versus wallet-balance requests
- `grounded_answer_from_supplied_facts`
- `unsupported_refusal`

This row is not admitted for:

- broad autonomous execution
- hidden tool-policy authority without bounded contracts
- unbounded long-context planning

## Validation Surface

There are two validation paths:

1. Contract validation
   - `scripts/check-compiled-agent-default-row-contract.sh`
   - verifies the committed contract fixture still matches the Rust generator
2. Live benchmark probe
   - `cargo run -q -p psionic-train --bin compiled_agent_default_row_probe -- ...`
   - measures the row on structured supported-intent routing, grounded-answer,
     and refusal cases over the OpenAI-compatible server surface

The live probe is intentionally small. It exists to freeze one honest latency,
quality, and refusal envelope for the first compiled-agent loop.
