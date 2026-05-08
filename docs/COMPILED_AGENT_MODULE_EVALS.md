# Compiled Agent Module Evals

> Status: canonical `AGENT-PLATFORM` independent module eval surface, updated 2026-03-28.

## Why This Exists

The first compiled-agent loop cannot be optimized honestly if route, tool
choice, tool arguments, grounded answer, and verification are only measured as
one blended end-to-end number.

This eval surface keeps those module families independent so GEPA, validator
gates, and later XTRAIN jobs can improve bounded behavior instead of rewarding
tool emission or general agent noise.

For Autopilot business behavior, Blueprint is the canonical program catalog.
These compiled-agent evals can inform Blueprint Module Versions or Optimizer
Runs, but they do not by themselves authorize CRM, CEO, Voice, Memory, legal,
source-system, or approval behavior. Blueprint owns Program Types, Program
Runs, promotion gates, evidence, and receipts for that layer.

## Canonical Report

The committed report lives at:

- `fixtures/compiled_agent/compiled_agent_module_eval_report_v1.json`

It is generated from:

- `crates/psionic-eval/src/compiled_agent_module_eval.rs`
- `crates/psionic-eval/examples/compiled_agent_module_eval_report.rs`

## What Is Measured Separately

- route selection
- tool-policy correctness
- tool-argument correctness
- grounded-answer correctness
- verify / fallback / unsupported refusal correctness

## Important Boundaries

- tool emission is not success
- unsupported refusal is measured separately from route selection
- negated prompts are retained explicitly as a learning surface
- a module can fail without contaminating the score of unrelated modules

## Current Baseline Truth

The baseline revision is intentionally narrow and leaves one known gap visible:

- explicit provider and wallet routes are green
- explicit tool-policy rows are green
- no-arg tool-call construction rows are green
- grounded-answer rows are green on the admitted narrow facts surface
- verify rows are green, including the guard that tool emission alone is not success
- negated wallet mention still false-positives on the route module

That last row is not hidden. It is retained so later bounded optimization can
prove improvement honestly.
