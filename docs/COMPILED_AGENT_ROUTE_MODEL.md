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
- training rows: `12`
- held-out rows: `6`

The model is trained from retained replay samples, not from hidden heuristics.
It converts the replay bundle into token and bigram counts, estimates per-route
feature likelihoods, and emits a stable JSON artifact that XTRAIN can validate
and promote.

## Current Retained Metrics

- training accuracy: `1.0`
- held-out accuracy: `1.0`
- artifact digest:
  `445fb8f77ca203d81476c935165b0c95f31ea4f6a85285872e6e0692268f5e22`

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

This artifact is retained, validator-scored, and already promoted inside the
Psionic-owned artifact contract.

It is still a narrow route model over a bounded task family. The right next
step is not broadening the task family, but continuing to grow the retained
receipt and held-out corpus so later route gains remain statistically and
operationally credible.
