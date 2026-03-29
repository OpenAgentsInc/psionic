# Compiled Agent Route Model

> Status: first replay-trained compiled-agent route model, updated 2026-03-29.

## What This Is

This is the first compiled-agent module candidate that is backed by a trained
artifact instead of a hand-authored rule delta.

The current scope stays deliberately narrow:

- module family: `route`
- task family: `provider_status` vs `wallet_status` vs `unsupported`
- training source: replay-backed route samples from
  `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`

## Model Shape

- family: `multinomial_naive_bayes`
- feature profile: `unigram_plus_bigram`
- artifact fixture: `fixtures/compiled_agent/compiled_agent_route_model_v1.json`

The model is trained from retained replay samples, not from hidden heuristics.
It converts the replay bundle into token and bigram counts, estimates per-route
feature likelihoods, and emits a stable JSON artifact that XTRAIN can validate
and promote.

## Why It Exists

The previous route candidate fixed the negated wallet failure with a bounded
keyword patch. That was useful, but it was not a real trained model.

The route model closes that gap:

- new receipts and replay samples can change the trained artifact
- the validator still gates promotion with independent module evals and replay
  matches
- the candidate can improve over time without pretending the whole agent has
  become self-modifying

## Entry Point

Regenerate the trained artifact and the XTRAIN outputs:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

## Honest Boundary

This artifact is retained and validator-scored inside `psionic` today.

It now appears in the retained compiled-agent promoted-artifact contract, but
that still does not by itself mean the `openagents` compiled-agent runtime is
already consuming it as live authority. Runtime adoption remains a separate
controlled step.
