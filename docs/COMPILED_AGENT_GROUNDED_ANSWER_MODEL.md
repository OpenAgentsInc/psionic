# Compiled Agent Grounded-Answer Model

> Status: first replay-trained grounded-answer model, updated 2026-03-29.

## What This Is

This is the first retained learned artifact for the compiled-agent
`grounded_answer` module.

The scope stays deliberately narrow:

- strict fact inputs only
- provider readiness answers
- wallet balance and recent earnings answers
- unsupported refusal
- fallback on missing facts
- fallback on conflicting facts

## Model Shape

- family: `multinomial_naive_bayes`
- feature profile: `route_plus_fact_signature`
- artifact fixture:
  `fixtures/compiled_agent/compiled_agent_grounded_answer_model_v1.json`

The model does not read the raw user prompt. It learns over the bounded route
and the supplied tool facts only.

## Current Retained Metrics

- training rows: `12`
- held-out rows: `6`
- training accuracy: `1.0`
- held-out accuracy: `1.0`
- artifact digest:
  `250dfa2deff1b02216195a3c40a4ff7865cf3b45e2f339d8c1f9679b47a0388b`

## Why This Matters

Before this change, grounded answer was still effectively a bounded rule
revision.

Now the learning loop retains a real grounded-answer artifact that:

- learns the supported answer programs from replay-backed grounded samples
- keeps unsupported refusal inside the retained module family
- falls back when provider or wallet facts are missing
- falls back when wallet facts conflict instead of picking one arbitrarily

## Independent Validator Surface

The independent module eval now includes:

- standard provider and wallet grounding rows
- unsupported refusal
- insufficient provider facts fallback
- conflicting wallet facts fallback

The current grounded candidate report passes all `6/6` grounded-answer eval
rows.

## Entry Point

Regenerate the learned grounded-answer artifact and XTRAIN outputs:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

## Honest Boundary

This is a real learned module, but it is still a narrow bounded surface. It
does not claim broad natural-language synthesis, hidden prompt intelligence, or
general chat-model behavior.
